from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1.  FrameBBoxBank – per-frame 3-D AABB storage
# ---------------------------------------------------------------------------

class FrameBBoxBank:
    """Maintains a 3-D axis-aligned bounding box (AABB) for each processed frame.

    When a frame is added, outlier points are removed along each axis using
    ``outlier_percentile`` before computing the tight bounding box.  Overlap
    queries then test the stored AABBs against the current frame's AABB.

    Args:
        outlier_percentile: Points outside the
            [p, 100-p] percentile range along each axis are discarded before
            computing the bounding box.  Default is ``2.0`` (removes the
            outermost 2 % on each side).
    """

    def __init__(self, outlier_percentile: float = 2.0) -> None:
        self.outlier_percentile = outlier_percentile
        # frame_idx -> (min_xyz, max_xyz), both shape (3,) CPU float tensors
        self._bboxes: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ── Internal helpers ────────────────────────────────────────────────────

    def _pts_to_bbox(
        self, pts: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute a robust AABB by discarding extreme-percentile outliers.

        Args:
            pts: ``(N, 3)`` float tensor on any device.

        Returns:
            ``(min_xyz, max_xyz)`` on CPU, or ``None`` if too few points remain.
        """
        pts = pts.float().cpu()
        if pts.shape[0] < 4:
            return None

        p = self.outlier_percentile / 100.0
        if p > 0:
            lo = torch.quantile(pts, p, dim=0)
            hi = torch.quantile(pts, 1.0 - p, dim=0)
            mask = ((pts >= lo) & (pts <= hi)).all(dim=1)
            pts = pts[mask]

        if pts.shape[0] == 0:
            return None

        return pts.min(dim=0).values, pts.max(dim=0).values

    @staticmethod
    def _aabb_intersects(
        min_a: torch.Tensor,
        max_a: torch.Tensor,
        min_b: torch.Tensor,
        max_b: torch.Tensor,
    ) -> bool:
        """Return ``True`` if two AABBs share a non-trivial volumetric overlap.

        The intersection volume must be at least 1 % of AABB *A*'s volume;
        degenerate (zero-volume) boxes always return ``False``.
        """
        has_intersection = (
            (min_a[0] <= max_b[0]) and (max_a[0] >= min_b[0])
            and (min_a[1] <= max_b[1]) and (max_a[1] >= min_b[1])
            and (min_a[2] <= max_b[2]) and (max_a[2] >= min_b[2])
        )
        if not bool(has_intersection):
            return False

        inter_min = torch.max(min_a, min_b)
        inter_max = torch.min(max_a, max_b)
        inter_vol = float((inter_max - inter_min).clamp(min=0.0).prod().item())

        a_vol = float((max_a - min_a).clamp(min=0.0).prod().item())
        if a_vol <= 0.0:
            return False

        return (inter_vol / a_vol) >= 0.01

    # ── Public API ───────────────────────────────────────────────────────────

    def add_frame(
        self,
        frame_idx: int,
        pts3d_patch: torch.Tensor,
        new_mask: torch.Tensor,
    ) -> None:
        """Compute and store the AABB of new-region points for a frame.

        Args:
            frame_idx: Index of the current frame.
            pts3d_patch: ``[B, ph, pw, 3]`` patch-grid point map (B=1).
            new_mask: ``[B, ph, pw]`` boolean mask of new-region patches.
        """
        pts = pts3d_patch[0].reshape(-1, 3)
        mask = new_mask[0].reshape(-1)
        pts = pts[mask].detach()

        result = self._pts_to_bbox(pts)
        if result is not None:
            self._bboxes[frame_idx] = result

    def get_bbox(
        self, frame_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return the stored AABB for *frame_idx*, or ``None`` if absent."""
        return self._bboxes.get(frame_idx, None)

    def frame_indices(self) -> List[int]:
        """Return all stored frame indices in ascending order."""
        return sorted(self._bboxes.keys())

    def find_overlapping_frames(self, query_pts: torch.Tensor) -> List[int]:
        """Return the indices of all frames whose AABB overlaps the query.

        Args:
            query_pts: ``(Q, 3)`` float tensor of points representing the
                current frame (outliers removed internally).

        Returns:
            Sorted list of frame indices with sufficient AABB overlap.
        """
        result = self._pts_to_bbox(query_pts)
        if result is None:
            return []
        cur_min, cur_max = result
        return [
            fidx
            for fidx, (fmin, fmax) in self._bboxes.items()
            if self._aabb_intersects(cur_min, cur_max, fmin, fmax)
        ]


# ---------------------------------------------------------------------------
# 2.  select_relevant_frame_indices
# ---------------------------------------------------------------------------

def select_relevant_frame_indices(
    current_frame_idx: int,
    all_processed_frame_indices: List[int],
    bbox_bank: FrameBBoxBank,
    current_pts3d_patch: torch.Tensor,
    first_k: int = 10,
    last_k: int = 10,
    neighbor_k: int = 5,
) -> Set[int]:
    """Determine which past frames should contribute KV tokens for the current frame.

    Three rules are combined:

    1. **First-K**: always include the earliest ``first_k`` processed frames.
    2. **Last-K**: always include the most recent ``last_k`` processed frames.
    3. **Spatial overlap ± neighbourhood**: for every past frame whose AABB
       overlaps the current frame's point cloud, include that frame and all
       frames within ±``neighbor_k`` indices.

    Args:
        current_frame_idx: Index of the frame being processed.
        all_processed_frame_indices: Ordered list of already-processed
            frame indices.
        bbox_bank: :class:`FrameBBoxBank` populated with past frames.
        current_pts3d_patch: ``[B, ph, pw, 3]`` approximate point map for the
            current frame (typically the previous frame's prediction).
        first_k: Number of earliest frames to always include.
        last_k: Number of most recent frames to always include.
        neighbor_k: Temporal radius around spatially overlapping frames.

    Returns:
        Set of frame indices to include in the KV context.
    """
    all_prev = [f for f in all_processed_frame_indices if f < current_frame_idx]
    if not all_prev:
        return set()

    keep: Set[int] = set()
    valid_set = set(all_prev)

    keep.update(all_prev[:first_k])
    keep.update(all_prev[-last_k:])

    query_pts = current_pts3d_patch[0].reshape(-1, 3).detach()
    if query_pts.shape[0] > 0:
        for rf in bbox_bank.find_overlapping_frames(query_pts):
            for nb in range(rf - neighbor_k, rf + neighbor_k + 1):
                if nb in valid_set:
                    keep.add(nb)

    return keep & valid_set


# ---------------------------------------------------------------------------
# 3.  filter_past_kv_by_frame_indices
# ---------------------------------------------------------------------------

def filter_past_kv_by_frame_indices(
    past_key_values: List,
    kv_img_idx: List[List[int]],
    keep_frame_indices: Set[int],
    target_device: Optional[torch.device] = None,
) -> Tuple[List, List[List[int]]]:
    """Extract KV tokens belonging to the specified frames from the cache.

    Args:
        past_key_values: Per-layer ``(k, v)`` cache, each tensor shaped
            ``[B, heads, seq, dim]``.
        kv_img_idx: Per-layer list mapping each sequence position to its
            originating frame index.
        keep_frame_indices: Set of frame indices whose tokens should be kept.
        target_device: If provided, move the selected tensors to this device.

    Returns:
        ``(filtered_past_kv, filtered_kv_idx)`` containing only the selected
        tokens.
    """
    keep_arr = np.array(sorted(keep_frame_indices), dtype=np.int64)
    filtered_past_kv = []
    filtered_kv_idx = []

    for kv, idx_list in zip(past_key_values, kv_img_idx):
        if kv is None or len(idx_list) == 0:
            filtered_past_kv.append(None)
            filtered_kv_idx.append([])
            continue

        k, v = kv
        idx_arr = np.array(idx_list, dtype=np.int64)
        keep_positions = np.where(np.isin(idx_arr, keep_arr))[0]

        if len(keep_positions) == 0:
            filtered_past_kv.append(None)
            filtered_kv_idx.append([])
            continue

        sel = torch.from_numpy(keep_positions).to(k.device)
        k_sel = k[:, :, sel, :]
        v_sel = v[:, :, sel, :]
        if target_device is not None and k_sel.device != target_device:
            k_sel = k_sel.to(target_device)
            v_sel = v_sel.to(target_device)
        filtered_past_kv.append((k_sel, v_sel))
        filtered_kv_idx.append(idx_arr[keep_positions].tolist())

    return filtered_past_kv, filtered_kv_idx
