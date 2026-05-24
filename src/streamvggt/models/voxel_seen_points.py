from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


class VoxelSeenPoints:
    """Voxel-downsampled set of previously observed 3-D points.

    Each voxel stores exactly one representative point; if a new observation
    falls into an already-occupied voxel the stored point is overwritten.
    When ``max_frames`` is set, voxels whose representative point is older
    than the window are evicted after each ``add_points`` call.

    Visibility is queried by checking the 3x3x3 neighbourhood of the
    candidate voxel:

    * Any neighbour occupied  → **seen**
    * No neighbour occupied   → **unseen**

    ``voxel_size`` therefore acts purely as a spatial discretisation scale.

    Args:
        voxel_size: Edge length of each voxel in world units.
        device: Torch device where point tensors are stored.
        max_voxels: Maximum number of voxels retained in the grid.
        max_frames: If set, evict voxels older than this many frames.
    """

    def __init__(
        self,
        voxel_size: float,
        device: torch.device,
        max_voxels: int = 500_000,
        max_frames: Optional[int] = None,
    ) -> None:
        self.voxel_size = voxel_size
        self.device = device
        self.max_voxels = max_voxels
        self.max_frames = max_frames

        self._voxel_to_idx: dict = {}
        self._points: Optional[torch.Tensor] = None   # (V, 3)
        self._frame_ids: Optional[torch.Tensor] = None  # (V,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_points(self, pts: torch.Tensor, frame_idx: int) -> None:
        """Insert new points into the voxel grid.

        Within the same call, only the last point that maps to a given voxel
        is retained.  For already-occupied voxels the representative point and
        frame-id are overwritten.

        Args:
            pts: ``(N, 3)`` float tensor of 3-D points.
            frame_idx: Integer index of the current frame.
        """
        if pts.shape[0] == 0:
            return

        pts = pts.float().to(self.device)
        vox = (pts / self.voxel_size).long()
        vox_np = vox.cpu().numpy()

        # Keep only the last point per voxel within this batch.
        key_to_last_i: dict = {}
        for i in range(pts.shape[0]):
            key = (int(vox_np[i, 0]), int(vox_np[i, 1]), int(vox_np[i, 2]))
            key_to_last_i[key] = i

        new_rows: List[torch.Tensor] = []
        new_keys: List[tuple] = []

        for key, i in key_to_last_i.items():
            if key in self._voxel_to_idx:
                idx = self._voxel_to_idx[key]
                self._points[idx] = pts[i]
                self._frame_ids[idx] = frame_idx
            else:
                if self._points is not None and len(self._voxel_to_idx) >= self.max_voxels:
                    continue
                new_keys.append(key)
                new_rows.append(pts[i])

        if new_rows:
            new_pts = torch.stack(new_rows, dim=0)
            start = 0 if self._points is None else self._points.shape[0]

            for j, key in enumerate(new_keys):
                self._voxel_to_idx[key] = start + j

            if self._points is None:
                self._points = new_pts
                self._frame_ids = torch.full(
                    (self._points.shape[0],),
                    frame_idx,
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                self._points = torch.cat([self._points, new_pts], dim=0)
                new_frame_ids = torch.full(
                    (new_pts.shape[0],),
                    frame_idx,
                    dtype=torch.long,
                    device=self.device,
                )
                self._frame_ids = torch.cat([self._frame_ids, new_frame_ids], dim=0)

        # Evict voxels that fall outside the temporal window.
        if self.max_frames is not None and self._frame_ids is not None:
            min_valid_frame = frame_idx - self.max_frames + 1
            keep_mask = self._frame_ids >= min_valid_frame
            if not torch.all(keep_mask):
                self._points = self._points[keep_mask]
                self._frame_ids = self._frame_ids[keep_mask]

                # Rebuild the voxel index from scratch.
                self._voxel_to_idx.clear()
                vox = (self._points / self.voxel_size).long().cpu().numpy()
                for idx, v in enumerate(vox):
                    key = (int(v[0]), int(v[1]), int(v[2]))
                    self._voxel_to_idx[key] = idx

    def has_seen_neighbors(self, pts_flat: torch.Tensor) -> torch.Tensor:
        """Check whether each query point has at least one occupied neighbour voxel.

        For every query point the 3x3x3 neighbourhood (27 voxels including
        the point's own voxel) is examined.  A point is classified as *seen*
        if any of those 27 voxels is occupied.

        Args:
            pts_flat: ``(Q, 3)`` float tensor of query points.

        Returns:
            Boolean tensor of shape ``(Q,)``; ``True`` means *seen*.
        """
        Q = pts_flat.shape[0]
        if self._points is None or self._points.shape[0] == 0:
            return torch.zeros((Q,), device=pts_flat.device, dtype=torch.bool)

        pts_flat = pts_flat.float().to(self.device)
        vox_q = (pts_flat / self.voxel_size).long()

        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]

        # Shape: (Q, 27, 3)
        neighbor_vox = torch.stack(
            [
                vox_q + torch.tensor(off, device=self.device, dtype=torch.long)
                for off in offsets
            ],
            dim=1,
        )

        neighbor_vox_np = neighbor_vox.reshape(-1, 3).cpu().numpy()
        unique_vox, inverse = np.unique(neighbor_vox_np, axis=0, return_inverse=True)

        unique_exists = np.zeros(len(unique_vox), dtype=bool)
        for i, row in enumerate(unique_vox):
            key = (int(row[0]), int(row[1]), int(row[2]))
            unique_exists[i] = key in self._voxel_to_idx

        exists_flat = unique_exists[inverse]
        exists = torch.from_numpy(exists_flat).to(self.device).reshape(Q, 27)

        return exists.any(dim=1)

    @property
    def num_points(self) -> int:
        """Number of representative points currently stored in the grid."""
        return 0 if self._points is None else self._points.shape[0]
