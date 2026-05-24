from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers.file_utils import ModelOutput

from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from streamvggt.models.aggregator import Aggregator
from streamvggt.models.kv_cache_utils import (
    move_kv_cache_to_device,
    replace_frame_kv_in_cache,
    update_kv_cache,
)
from streamvggt.models.voxel_seen_points import VoxelSeenPoints


def _pts3d_to_patch_grid(
    pts3d: torch.Tensor,
    patch_size: int,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    B, h, w, _ = pts3d.shape
    if h == patch_h and w == patch_w:
        return pts3d
    x = pts3d.permute(0, 3, 1, 2)  # [B, 3, H, W]
    x = torch.nn.functional.adaptive_avg_pool2d(x, (patch_h, patch_w))
    return x.permute(0, 2, 3, 1)   # [B, patch_h, patch_w, 3]


# ---------------------------------------------------------------------------
# New-region detection
# ---------------------------------------------------------------------------

def compute_new_region_mask(
    pts3d: torch.Tensor,
    pts3d_conf: torch.Tensor,
    seen_points_3d: Optional[Union[torch.Tensor, VoxelSeenPoints]],
    patch_size: int,
    patch_h: int,
    patch_w: int,
    dist_threshold: float = 0.05,
    conf_threshold: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = pts3d.shape[0]
    pts3d_patch = _pts3d_to_patch_grid(pts3d, patch_size, patch_h, patch_w)

    # Downsample confidence to the patch grid.
    if pts3d_conf.dim() == 2:
        pts3d_conf = pts3d_conf.unsqueeze(0)
    conf_patch = torch.nn.functional.adaptive_avg_pool2d(
        pts3d_conf.unsqueeze(1), (patch_h, patch_w)
    ).squeeze(1)

    num_patches = patch_h * patch_w
    pts_flat = pts3d_patch.reshape(B * num_patches, 3)

    is_empty = seen_points_3d is None or (
        isinstance(seen_points_3d, VoxelSeenPoints) and seen_points_3d.num_points == 0
    ) or (
        isinstance(seen_points_3d, torch.Tensor) and seen_points_3d.shape[0] == 0
    )
    if is_empty:
        new_mask = torch.ones(B, patch_h, patch_w, dtype=torch.bool, device=pts3d.device)
        return new_mask, pts3d_patch

    with torch.no_grad():
        if isinstance(seen_points_3d, VoxelSeenPoints):
            seen_mask = seen_points_3d.has_seen_neighbors(pts_flat)
        else:
            dists = torch.cdist(pts_flat.float(), seen_points_3d.float())
            min_dist = dists.min(dim=1).values
            seen_mask = min_dist < dist_threshold

        seen_mask = seen_mask.reshape(B, patch_h, patch_w)
        seen_region = seen_mask & (conf_patch >= conf_threshold)
        new_mask = ~seen_region

    return new_mask, pts3d_patch


# ---------------------------------------------------------------------------
# KV-list pruning by patch mask
# ---------------------------------------------------------------------------

def filter_kv_list_by_patch_mask(
    new_kv_list: List,
    patch_start_idx: int,
    new_mask: torch.Tensor,
) -> List:
    B, patch_h, patch_w = new_mask.shape
    if B > 1:
        raise NotImplementedError("filter_kv_list_by_patch_mask only supports B=1")

    num_patches = patch_h * patch_w
    keep_flat = new_mask.reshape(B, num_patches)

    special_indices = torch.arange(patch_start_idx, device=new_mask.device)
    patch_indices = torch.where(keep_flat[0])[0] + patch_start_idx
    keep_indices = torch.cat([special_indices, patch_indices], dim=0)

    return [
        (k[:, :, keep_indices, :], v[:, :, keep_indices, :])
        for k, v in new_kv_list
    ]


@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class StreamVGGT(nn.Module, PyTorchModelHubMixin):

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        total_budget: int = 1_200_000,
    ) -> None:
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim
        )
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=4,
            activation="inv_log",
            conf_activation="expp1",
        )
        self.depth_head = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
        )
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.total_budget = total_budget

    # ------------------------------------------------------------------
    # Standard forward (batch mode, no KV cache)
    # ------------------------------------------------------------------

    def forward(
        self,
        views: List[dict],
        query_points: Optional[torch.Tensor] = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache: bool = False,
        past_frame_idx: int = 0,
    ) -> StreamVGGTOutput:
        images = torch.stack(
            [view["img"] for view in views], dim=0
        ).permute(1, 0, 2, 3, 4)  # [B, S, C, H, W]

        if images.ndim == 4:
            images = images.unsqueeze(0)
        if query_points is not None and query_points.ndim == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions: dict = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    query_points=query_points,
                )
                predictions["track"] = track_list[-1]
                predictions["vis"] = vis
                predictions["conf"] = conf

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    "pts3d_in_other_view": predictions["world_points"][:, s],   # [B, H, W, 3]
                    "conf": predictions["world_points_conf"][:, s],             # [B, H, W]
                    "depth": predictions["depth"][:, s],                        # [B, H, W, 1]
                    "depth_conf": predictions["depth_conf"][:, s],              # [B, H, W]
                    "camera_pose": predictions["pose_enc"][:, s, :],            # [B, 9]
                    **({
                        "valid_mask": views[s]["valid_mask"]
                    } if "valid_mask" in views[s] else {}),
                    **({
                        "track": predictions["track"][:, s],      # [B, N, 2]
                        "vis": predictions["vis"][:, s],           # [B, N]
                        "track_conf": predictions["conf"][:, s],
                    } if "track" in predictions else {}),
                }
                ress.append(res)

        return StreamVGGTOutput(ress=ress, views=views)

    # ------------------------------------------------------------------
    # Streaming inference (single-frame at a time, with KV cache)
    # ------------------------------------------------------------------

    def inference(
        self,
        frames,
        query_points: Optional[torch.Tensor] = None,
        frame_writer: Optional[Callable] = None,
        cache_results: bool = True,
        filter_new_region: bool = True,
        new_region_dist_threshold: float = 0.02,
        new_region_conf_threshold: float = 0.3,
        max_seen_points: int = 100_000_000,
        # Loop-closure parameters
        loop_closure_first_k: int = 5,
        loop_closure_last_k: int = 1,
        loop_closure_neighbor_k: int = 21,
        loop_closure_outlier_pct: float = 10.0,
        max_frames: int = 20,
    ) -> StreamVGGTOutput:
        """Process a sequence of frames one at a time with KV caching.

        Loop-closure-aware frame selection keeps only spatially relevant past
        KV tokens, preventing memory from growing unboundedly.

        Args:
            frames: Iterable of frame dicts, each containing at least an
                ``"img"`` tensor of shape ``[C, H, W]`` and optionally a
                ``"valid_mask"`` tensor.
            query_points: Optional ``[1, N, 2]`` initial track query points.
                Updated in-place as tracking propagates forward.
            frame_writer: Optional callback ``(frame_idx, frame, result)``
                called after each frame is processed.
            cache_results: If ``True``, accumulate all results in memory and
                return them in the output object.
            filter_new_region: Whether to prune KV tokens that correspond to
                already-seen 3-D regions.
            new_region_dist_threshold: Voxel size used for new-region
                detection (world units).
            new_region_conf_threshold: Minimum point confidence to consider
                a patch as *seen*.
            max_seen_points: Maximum number of voxels in the seen-point set.
            loop_closure_first_k: Always include the first *k* frames.
            loop_closure_last_k: Always include the most recent *k* frames.
            loop_closure_neighbor_k: Temporal neighbourhood radius around
                each spatially overlapping frame.
            loop_closure_outlier_pct: Percentile used to remove outlier
                points when computing per-frame bounding boxes.
            max_frames: Temporal window for the voxel seen-point set.

        Returns:
            :class:`StreamVGGTOutput` with accumulated results (when
            ``cache_results=True``).
        """
        from tqdm import tqdm

        from streamvggt.models.loop_closure_kv import (
            FrameBBoxBank,
            filter_past_kv_by_frame_indices,
            select_relevant_frame_indices,
        )

        # ── Initialise caches ────────────────────────────────────────────────
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        total_budget = self.total_budget
        seen_points_3d: Optional[VoxelSeenPoints] = None
        patch_size = self.aggregator.patch_size
        patch_start_idx = self.aggregator.patch_start_idx

        all_ress: List[dict] = []
        processed_frames: List[dict] = []
        kv_img_idx = [[] for _ in range(self.aggregator.depth)]
        kv_img_idx_camera = [[] for _ in range(self.camera_head.trunk_depth)]

        # ── Loop-closure bookkeeping ─────────────────────────────────────────
        bbox_bank = FrameBBoxBank(outlier_percentile=loop_closure_outlier_pct)
        all_processed_frame_indices: List[int] = []
        last_pts3d_patch: Optional[torch.Tensor] = None

        # All frames are immediately pruned (no full-token retention window).
        RECENT_FULL_FRAMES = 0
        recent_full_kv_buffer: List[Tuple[int, List]] = []

        # ── Main loop ────────────────────────────────────────────────────────
        for i, frame in tqdm(enumerate(frames)):
            images = frame["img"].unsqueeze(0)  # [1, C, H, W]

            # Select which past frames to include in the KV context.
            if i > 0 and last_pts3d_patch is not None:
                relevant_indices = select_relevant_frame_indices(
                    current_frame_idx=i,
                    all_processed_frame_indices=all_processed_frame_indices,
                    bbox_bank=bbox_bank,
                    current_pts3d_patch=last_pts3d_patch,
                    first_k=loop_closure_first_k,
                    last_k=loop_closure_last_k,
                    neighbor_k=loop_closure_neighbor_k,
                )
                agg_past_kv, _ = filter_past_kv_by_frame_indices(
                    past_key_values,
                    kv_img_idx,
                    relevant_indices,
                    target_device=images.device,
                )
            else:
                agg_past_kv = move_kv_cache_to_device(past_key_values, images.device)

            # ── Aggregator forward ───────────────────────────────────────────
            aggregator_output = self.aggregator(
                images,
                past_key_values=agg_past_kv,
                use_cache=True,
                past_frame_idx=i,
                total_budget=total_budget,
            )
            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, new_kv_list = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output

            new_kv_list_to_cache = new_kv_list

            with torch.cuda.amp.autocast(enabled=False):
                # Camera head
                if self.camera_head is not None:
                    pose_enc, block_kv_list = self.camera_head(
                        aggregated_tokens,
                        past_key_values_camera=past_key_values_camera,
                        use_cache=True,
                    )
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                # Depth head
                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens,
                        images=images,
                        patch_start_idx=patch_start_idx,
                    )
                    depth = depth[:, 0]
                    depth_conf = depth_conf[:, 0]

                # Point head
                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens,
                        images=images,
                        patch_start_idx=patch_start_idx,
                    )
                    pts3d = pts3d[:, 0]
                    pts3d_conf = pts3d_conf[:, 0]

                # New-region filtering
                if filter_new_region and self.point_head is not None:
                    H, W = images.shape[-2], images.shape[-1]
                    ph, pw = H // patch_size, W // patch_size
                    new_mask, pts3d_patch = compute_new_region_mask(
                        pts3d,
                        pts3d_conf,
                        seen_points_3d,
                        patch_size,
                        ph,
                        pw,
                        dist_threshold=new_region_dist_threshold,
                        conf_threshold=new_region_conf_threshold,
                    )
                    new_kv_list_to_cache = filter_kv_list_by_patch_mask(
                        new_kv_list, patch_start_idx, new_mask
                    )

                    new_pts = pts3d_patch.reshape(-1, 3)[new_mask.reshape(-1)]
                    if new_pts.shape[0] > 0:
                        new_pts = new_pts.detach().float()
                        if seen_points_3d is None:
                            seen_points_3d = VoxelSeenPoints(
                                voxel_size=new_region_dist_threshold,
                                device=new_pts.device,
                                max_voxels=max_seen_points,
                                max_frames=max_frames,
                            )
                        seen_points_3d.add_points(new_pts, frame_idx=i)

                    bbox_bank.add_frame(i, pts3d_patch, new_mask)
                    last_pts3d_patch = pts3d_patch.detach()

                # Tracking head
                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens,
                        images=images,
                        patch_start_idx=patch_start_idx,
                        query_points=query_points,
                    )
                    track = track_list[-1][:, 0]
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            # ── Update KV caches ─────────────────────────────────────────────
            past_key_values, kv_img_idx = update_kv_cache(
                past_key_values, new_kv_list_to_cache, kv_img_idx, i
            )

            # Store pruned tokens on CPU and immediately replace in the cache
            # so the in-GPU cache only ever holds the pruned representation.
            pruned_kv_cpu = [
                (k.detach().cpu(), v.detach().cpu())
                for k, v in new_kv_list_to_cache
            ]
            recent_full_kv_buffer.append((i, pruned_kv_cpu))

            if len(recent_full_kv_buffer) > RECENT_FULL_FRAMES:
                old_frame_idx, old_pruned_kv = recent_full_kv_buffer.pop(0)
                past_key_values, kv_img_idx = replace_frame_kv_in_cache(
                    past_key_values, kv_img_idx, old_frame_idx, old_pruned_kv
                )

            past_key_values_camera, kv_img_idx_camera = update_kv_cache(
                past_key_values_camera, block_kv_list, kv_img_idx_camera, i
            )

            all_processed_frame_indices.append(i)

            # ── Collect results ──────────────────────────────────────────────
            res_gpu = {
                "pts3d_in_other_view": pts3d,
                "conf": pts3d_conf,
                "depth": depth,
                "depth_conf": depth_conf,
                "camera_pose": camera_pose,
                **({"valid_mask": frame["valid_mask"]} if "valid_mask" in frame else {}),
                **(
                    {"track": track, "vis": vis, "track_conf": track_conf}
                    if query_points is not None
                    else {}
                ),
            }
            res_cpu = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in res_gpu.items()
            }

            if frame_writer is not None:
                frame_writer(i, frame, res_cpu)
            if cache_results:
                all_ress.append(res_cpu)
                processed_frames.append(
                    {
                        nk: nv.detach().cpu() if isinstance(nv, torch.Tensor) else nv
                        for nk, nv in frame.items()
                    }
                )

            del res_gpu
            torch.cuda.empty_cache()

        return StreamVGGTOutput(
            ress=all_ress if cache_results else None,
            views=processed_frames if cache_results else None,
        )
