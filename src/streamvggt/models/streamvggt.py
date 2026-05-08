import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from streamvggt.models.aggregator import Aggregator
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Any, Callable, Union
from dataclasses import dataclass
import numpy as np


def replace_frame_kv_in_cache(
    past_key_values: List,
    kv_img_idx: List,
    frame_idx: int,
    replacement_kv_list: List,
) -> Tuple[List, List]:
    """
    将 cache 中某一帧对应的 token 全量替换为 replacement_kv_list（通常为剪枝后的 token）。
    要求 kv_img_idx 中同一帧 token 在时序上连续（由 append 逻辑保证）。
    """
    for layer_idx in range(len(past_key_values)):
        if past_key_values[layer_idx] is None:
            continue
        k, v = past_key_values[layer_idx]
        img_idx_list = kv_img_idx[layer_idx]
        positions = [pos for pos, fid in enumerate(img_idx_list) if fid == frame_idx]
        if not positions:
            continue

        start = positions[0]
        end = positions[-1] + 1
        rep_k, rep_v = replacement_kv_list[layer_idx]
        rep_k = rep_k.to(k.device)
        rep_v = rep_v.to(v.device)
        # print("replace_frame_kv_in_cache: start =", start, "end =", end, "len=", end-start, rep_k.shape, rep_v.shape)
        new_k = torch.cat([k[:, :, :start, :], rep_k, k[:, :, end:, :]], dim=2)
        new_v = torch.cat([v[:, :, :start, :], rep_v, v[:, :, end:, :]], dim=2)
        past_key_values[layer_idx] = (new_k, new_v)

        old_len = end - start
        new_len = rep_k.shape[2]
        kv_img_idx[layer_idx] = (
            img_idx_list[:start]
            + [frame_idx] * new_len
            + img_idx_list[start + old_len:]
        )
    return past_key_values, kv_img_idx

def update_kv_cache(past_key_values: List, new_kv_list: List, kv_img_idx: List, img_idx: int) -> List:
    """将 new_kv_list 拼接到 past_key_values，无则直接赋值。"""
    for idx in range(len(new_kv_list)):
        if past_key_values[idx] is not None:
            past_k, past_v = past_key_values[idx]
            k, v = new_kv_list[idx]
            new_k = torch.cat([past_k, k.to(past_k.device)], dim=2)
            new_v = torch.cat([past_v, v.to(past_v.device)], dim=2)
            past_key_values[idx] = (new_k, new_v)
            
        else:
            past_key_values[idx] = new_kv_list[idx]
        kv_img_idx[idx] += [img_idx] * new_kv_list[idx][0].shape[2]
    return past_key_values, kv_img_idx


def move_kv_cache_to_device(past_key_values: List, device: torch.device) -> List:
    """
    将 past_key_values 中的所有 KV 移动到指定 device（保持结构不变）。
    """
    if past_key_values is None:
        return None
    for idx, kv in enumerate(past_key_values):
        if kv is None:
            continue
        k, v = kv
        if k.device != device:
            past_key_values[idx] = (k.to(device), v.to(device))
    return past_key_values


def _pts3d_to_patch_grid(
    pts3d: torch.Tensor, patch_size: int, patch_h: int, patch_w: int
) -> torch.Tensor:
    """将 DPT 输出的 [B, H', W', 3] 下采样到 patch 网格 [B, patch_h, patch_w, 3]。"""
    B, h, w, _ = pts3d.shape
    if h == patch_h and w == patch_w:
        return pts3d
    # [B, H', W', 3] -> [B, 3, H', W']
    x = pts3d.permute(0, 3, 1, 2)
    x = torch.nn.functional.adaptive_avg_pool2d(x, (patch_h, patch_w))
    return x.permute(0, 2, 3, 1)



# class VoxelSeenPoints:
#     """
#     体素降采样的已见 3D 点集合：每个体素只保留一个代表点（同一体素内始终用**最新帧**的
#     观测覆盖）；若设置 max_frames，则只保留最近若干帧内的代表点，久未落入点的体素会随
#     裁剪被移除。查询时只与当前点所在体素及 26 邻域体素内的代表点比较，将复杂度从
#     O(N*M) 降为 O(N*27)。
#     """

#     def __init__(
#         self,
#         voxel_size: float,
#         device: torch.device,
#         max_voxels: int = 500000,
#         max_frames: Optional[int] = None,
#     ):
#         self.voxel_size = voxel_size
#         self.device = device
#         self.max_voxels = max_voxels
#         self._voxel_to_idx: dict = {}
#         self._points: Optional[torch.Tensor] = None  # (V, 3)
#         self.max_frames = max_frames
#         self._frame_ids: Optional[torch.Tensor] = None  # (V,)

#     def add_points(self, pts: torch.Tensor, frame_idx: int) -> None:
#         """将新点加入体素网格；同一体素内用当前帧的最新观测覆盖已有代表点。"""
#         if pts.shape[0] == 0:
#             return
#         pts = pts.float().to(self.device)
#         vox = (pts / self.voxel_size).long()
#         vox_np = vox.cpu().numpy()
#         # 本帧内同一体素只保留最后一次落入该体素的点
#         key_to_last_i: dict = {}
#         for i in range(pts.shape[0]):
#             key = (int(vox_np[i, 0]), int(vox_np[i, 1]), int(vox_np[i, 2]))
#             key_to_last_i[key] = i

#         new_rows: List[torch.Tensor] = []
#         new_keys: List[tuple] = []
#         for key, i in key_to_last_i.items():
#             if key in self._voxel_to_idx:
#                 idx = self._voxel_to_idx[key]
#                 self._points[idx] = pts[i]
#                 self._frame_ids[idx] = frame_idx
#             else:
#                 if self._points is not None and len(self._voxel_to_idx) >= self.max_voxels:
#                     continue
#                 new_keys.append(key)
#                 new_rows.append(pts[i])

#         if new_rows:
#             new_pts = torch.stack(new_rows, dim=0)
#             start = 0 if self._points is None else self._points.shape[0]
#             for j, key in enumerate(new_keys):
#                 self._voxel_to_idx[key] = start + j
#             if self._points is None:
#                 self._points = new_pts
#                 self._frame_ids = torch.full(
#                     (self._points.shape[0],),
#                     frame_idx,
#                     dtype=torch.long,
#                     device=self.device,
#                 )
#             else:
#                 self._points = torch.cat([self._points, new_pts], dim=0)
#                 new_frame_ids = torch.full(
#                     (new_pts.shape[0],),
#                     frame_idx,
#                     dtype=torch.long,
#                     device=self.device,
#                 )
#                 self._frame_ids = torch.cat([self._frame_ids, new_frame_ids], dim=0)

#         # 只保留最近 max_frames 帧的点
#         if self.max_frames is not None and self._frame_ids is not None:
#             min_valid_frame = frame_idx - self.max_frames + 1
#             keep_mask = self._frame_ids >= min_valid_frame
#             if not torch.all(keep_mask):
#                 self._points = self._points[keep_mask]
#                 self._frame_ids = self._frame_ids[keep_mask]
#                 # 重新构建体素索引
#                 self._voxel_to_idx.clear()
#                 vox = (self._points / self.voxel_size).long().cpu().numpy()
#                 for idx, v in enumerate(vox):
#                     key = (int(v[0]), int(v[1]), int(v[2]))
#                     self._voxel_to_idx[key] = idx

#     def min_distance_to_seen(self, pts_flat: torch.Tensor) -> torch.Tensor:
#         """
#         对每个查询点，只在与该点所在体素及 26 邻域体素内的代表点中求最小距离。
#         pts_flat: (Q, 3)
#         Returns: (Q,) 每个点到已见点的最小距离，若无邻域代表点则为 inf。
#         """
#         Q = pts_flat.shape[0]
#         if self._points is None or self._points.shape[0] == 0:
#             return torch.full((Q,), float("inf"), device=pts_flat.device, dtype=pts_flat.dtype)
#         pts_flat = pts_flat.float().to(self.device)
#         vox_q = (pts_flat / self.voxel_size).long()
#         # 27 邻域偏移
#         offsets = [
#             (dx, dy, dz)
#             for dx in (-1, 0, 1)
#             for dy in (-1, 0, 1)
#             for dz in (-1, 0, 1)
#         ]
#         # (Q*27, 3) 所有查询点的邻域体素坐标
#         neighbor_vox = torch.stack(
#             [vox_q + torch.tensor(off, device=self.device, dtype=torch.long) for off in offsets],
#             dim=1,
#         ).reshape(-1, 3)
#         neighbor_vox_np = neighbor_vox.cpu().numpy()
#         unique_vox, inverse = np.unique(neighbor_vox_np, axis=0, return_inverse=True)
#         unique_indices = np.full(len(unique_vox), -1, dtype=np.int64)
#         for i, row in enumerate(unique_vox):
#             key = (int(row[0]), int(row[1]), int(row[2]))
#             unique_indices[i] = self._voxel_to_idx.get(key, -1)
#         indices_flat = unique_indices[inverse]
#         indices = torch.from_numpy(indices_flat).to(self.device).reshape(Q, 27)
#         min_dist = torch.full(
#             (Q,), float("inf"), device=self.device, dtype=pts_flat.dtype
#         )
#         for j in range(27):
#             idx = indices[:, j]
#             valid = idx >= 0
#             if valid.any():
#                 d = torch.full((Q,), float("inf"), device=self.device, dtype=pts_flat.dtype)
#                 d[valid] = torch.norm(
#                     pts_flat[valid] - self._points[idx[valid]], dim=1
#                 )
#                 min_dist = torch.minimum(min_dist, d)
#         return min_dist

#     @property
#     def num_points(self) -> int:
#         return 0 if self._points is None else self._points.shape[0]

class VoxelSeenPoints:
    """
    体素降采样的已见 3D 点集合：每个体素只保留一个代表点（同一体素内始终用最新帧的
    观测覆盖）；若设置 max_frames，则只保留最近若干帧内的代表点，久未落入点的体素会随
    裁剪被移除。

    查询时不再计算距离，也不再使用额外阈值，而是只检查当前点所在体素及 26 邻域体素中
    是否存在已见点：
        - 只要邻域内存在任意点，则判为 seen
        - 若 27 个体素内都没有点，则判为 unseen

    因此 voxel_size 仅用于定义空间离散粒度。
    """

    def __init__(
        self,
        voxel_size: float,
        device: torch.device,
        max_voxels: int = 500000,
        max_frames: Optional[int] = None,
    ):
        self.voxel_size = voxel_size
        self.device = device
        self.max_voxels = max_voxels
        self._voxel_to_idx: dict = {}
        self._points: Optional[torch.Tensor] = None  # (V, 3)
        self.max_frames = max_frames
        self._frame_ids: Optional[torch.Tensor] = None  # (V,)

    def add_points(self, pts: torch.Tensor, frame_idx: int) -> None:
        """将新点加入体素网格；同一体素内用当前帧的最新观测覆盖已有代表点。"""
        if pts.shape[0] == 0:
            return

        pts = pts.float().to(self.device)
        vox = (pts / self.voxel_size).long()
        vox_np = vox.cpu().numpy()

        # 本帧内同一体素只保留最后一次落入该体素的点
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

        # 只保留最近 max_frames 帧的点
        if self.max_frames is not None and self._frame_ids is not None:
            min_valid_frame = frame_idx - self.max_frames + 1
            keep_mask = self._frame_ids >= min_valid_frame
            if not torch.all(keep_mask):
                self._points = self._points[keep_mask]
                self._frame_ids = self._frame_ids[keep_mask]

                # 重新构建体素索引
                self._voxel_to_idx.clear()
                vox = (self._points / self.voxel_size).long().cpu().numpy()
                for idx, v in enumerate(vox):
                    key = (int(v[0]), int(v[1]), int(v[2]))
                    self._voxel_to_idx[key] = idx

    def has_seen_neighbors(self, pts_flat: torch.Tensor) -> torch.Tensor:
        """
        对每个查询点，只检查其所在体素及 26 邻域体素中是否存在已见点。
        pts_flat: (Q, 3)

        Returns:
            seen_mask: (Q,) bool
                True  -> 邻域内存在已见点，判为 seen
                False -> 邻域内不存在点，判为 unseen
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

        # (Q, 27, 3)
        neighbor_vox = torch.stack(
            [vox_q + torch.tensor(off, device=self.device, dtype=torch.long) for off in offsets],
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

        seen_mask = exists.any(dim=1)
        return seen_mask

    @property
    def num_points(self) -> int:
        return 0 if self._points is None else self._points.shape[0]
class VoxelSeenPoints_v2:
    """
    倒叙
    体素降采样的已见 3D 点集合：每个体素只保留一个代表点。若设置 max_frames，则只保留最近
    若干帧内的代表点；**在体素仍有效时**新帧观测不会覆盖该体素已有代表点，待该点超出
    时间窗被裁剪后，后续帧才可重新填入该体素。未设置 max_frames 时行为与原先一致：同一体素
    内用当前帧的最新观测覆盖。查询时只与当前点所在体素及 26 邻域体素内的代表点比较，将
    复杂度从 O(N*M) 降为 O(N*27)。
    """

    def __init__(
        self,
        voxel_size: float,
        device: torch.device,
        max_voxels: int = 500000,
        max_frames: Optional[int] = None,
    ):
        self.voxel_size = voxel_size
        self.device = device
        self.max_voxels = max_voxels
        self._voxel_to_idx: dict = {}
        self._points: Optional[torch.Tensor] = None  # (V, 3)
        self.max_frames = max_frames
        self._frame_ids: Optional[torch.Tensor] = None  # (V,)

    def add_points(self, pts: torch.Tensor, frame_idx: int) -> None:
        """将新点加入体素网格。max_frames 为 None 时同一体素用当前帧最新观测覆盖；否则先按
        时间窗裁剪腾空过期体素，且**不覆盖**仍有效的体素代表点。"""
        if pts.shape[0] == 0:
            return
        pts = pts.float().to(self.device)
        # 先裁剪，使超出 max_frames 的代表点（及对应体素槽位）被释放，供本帧重新填入
        if self.max_frames is not None and self._frame_ids is not None:
            min_valid_frame = frame_idx - self.max_frames + 1
            keep_mask = self._frame_ids >= min_valid_frame
            if not torch.all(keep_mask):
                self._points = self._points[keep_mask]
                self._frame_ids = self._frame_ids[keep_mask]
                self._voxel_to_idx.clear()
                vox_kept = (self._points / self.voxel_size).long().cpu().numpy()
                for idx, v in enumerate(vox_kept):
                    key = (int(v[0]), int(v[1]), int(v[2]))
                    self._voxel_to_idx[key] = idx

        vox = (pts / self.voxel_size).long()
        vox_np = vox.cpu().numpy()
        # 本帧内同一体素只保留最后一次落入该体素的点
        key_to_last_i: dict = {}
        for i in range(pts.shape[0]):
            key = (int(vox_np[i, 0]), int(vox_np[i, 1]), int(vox_np[i, 2]))
            key_to_last_i[key] = i

        new_rows: List[torch.Tensor] = []
        new_keys: List[tuple] = []
        for key, i in key_to_last_i.items():
            if key in self._voxel_to_idx:
                if self.max_frames is None:
                    idx = self._voxel_to_idx[key]
                    self._points[idx] = pts[i]
                    self._frame_ids[idx] = frame_idx
                # max_frames 已设置：体素仍占用则不覆盖，等时间窗淘汰后再由后续帧写入
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

    def min_distance_to_seen(self, pts_flat: torch.Tensor) -> torch.Tensor:
        """
        对每个查询点，只在与该点所在体素及 26 邻域体素内的代表点中求最小距离。
        pts_flat: (Q, 3)
        Returns: (Q,) 每个点到已见点的最小距离，若无邻域代表点则为 inf。
        """
        Q = pts_flat.shape[0]
        if self._points is None or self._points.shape[0] == 0:
            return torch.full((Q,), float("inf"), device=pts_flat.device, dtype=pts_flat.dtype)
        pts_flat = pts_flat.float().to(self.device)
        vox_q = (pts_flat / self.voxel_size).long()
        # 27 邻域偏移
        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]
        # (Q*27, 3) 所有查询点的邻域体素坐标
        neighbor_vox = torch.stack(
            [vox_q + torch.tensor(off, device=self.device, dtype=torch.long) for off in offsets],
            dim=1,
        ).reshape(-1, 3)
        neighbor_vox_np = neighbor_vox.cpu().numpy()
        unique_vox, inverse = np.unique(neighbor_vox_np, axis=0, return_inverse=True)
        unique_indices = np.full(len(unique_vox), -1, dtype=np.int64)
        for i, row in enumerate(unique_vox):
            key = (int(row[0]), int(row[1]), int(row[2]))
            unique_indices[i] = self._voxel_to_idx.get(key, -1)
        indices_flat = unique_indices[inverse]
        indices = torch.from_numpy(indices_flat).to(self.device).reshape(Q, 27)
        min_dist = torch.full(
            (Q,), float("inf"), device=self.device, dtype=pts_flat.dtype
        )
        for j in range(27):
            idx = indices[:, j]
            valid = idx >= 0
            if valid.any():
                d = torch.full((Q,), float("inf"), device=self.device, dtype=pts_flat.dtype)
                d[valid] = torch.norm(
                    pts_flat[valid] - self._points[idx[valid]], dim=1
                )
                min_dist = torch.minimum(min_dist, d)
        return min_dist

    @property
    def num_points(self) -> int:
        return 0 if self._points is None else self._points.shape[0]



def compute_new_region_mask(
    pts3d: torch.Tensor,
    pts3d_conf: torch.Tensor,
    seen_points_3d: Optional[Union[torch.Tensor, VoxelSeenPoints, VoxelSeenPoints_v2]],
    patch_size: int,
    patch_h: int,
    patch_w: int,
    dist_threshold: float = 0.05,
    conf_threshold: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据本帧 3D 点与历史已见 3D 点，判断每个 patch 是否为新见区域。
    仅当 conf 足够高且 3D 点与历史某点距离 < dist_threshold 时视为「已见过」。
    seen_points_3d 可为 VoxelSeenPoints（体素降采样，查询快）或 Tensor（兼容旧逻辑）。
    Returns:
        new_mask: [B, patch_h, patch_w] bool，True 表示新见区域（保留该 token）
        pts3d_patch: [B, patch_h, patch_w, 3]，用于更新 seen_points_3d
    """
    import time
    start_time = time.time()
    B = pts3d.shape[0]
    pts3d_patch = _pts3d_to_patch_grid(pts3d, patch_size, patch_h, patch_w)
    # conf 下采样到同一网格
    if pts3d_conf.dim() == 2:
        pts3d_conf = pts3d_conf.unsqueeze(0)
    conf_patch = torch.nn.functional.adaptive_avg_pool2d(
        pts3d_conf.unsqueeze(1), (patch_h, patch_w)
    ).squeeze(1)

    num_patches = pts3d_patch.shape[1] * pts3d_patch.shape[2]
    pts_flat = pts3d_patch.reshape(B * num_patches, 3)

    is_empty = seen_points_3d is None or (
        (isinstance(seen_points_3d, VoxelSeenPoints) or isinstance(seen_points_3d, VoxelSeenPoints_v2)) and seen_points_3d.num_points == 0
    ) or (isinstance(seen_points_3d, torch.Tensor) and seen_points_3d.shape[0] == 0)
    if is_empty:
        new_mask = torch.ones(B, patch_h, patch_w, dtype=torch.bool, device=pts3d.device)
        return new_mask, pts3d_patch

    with torch.no_grad():
        if isinstance(seen_points_3d, VoxelSeenPoints) or isinstance(seen_points_3d, VoxelSeenPoints_v2):
            # min_dist = seen_points_3d.min_distance_to_seen(pts_flat)
            seen_mask = seen_points_3d.has_seen_neighbors(pts_flat)
        else:
            dists = torch.cdist(pts_flat.float(), seen_points_3d.float())
            min_dist = dists.min(dim=1).values
        # min_dist = min_dist.reshape(B, patch_h, patch_w)
        seen_mask = seen_mask.reshape(B, patch_h, patch_w)
        # 已见过：conf 高且距离近
        # seen_region = (conf_patch >= conf_threshold) & (min_dist < dist_threshold)
        seen_region = (seen_mask) & (conf_patch >= conf_threshold)
        new_mask = ~seen_region
    end_time = time.time()
    # print(f"compute_new_region_mask time: {end_time - start_time} seconds")
    return new_mask, pts3d_patch


def filter_kv_list_by_patch_mask(
    new_kv_list: List,
    patch_start_idx: int,
    new_mask: torch.Tensor,
) -> List:
    """
    只保留 special tokens (0:patch_start_idx) 以及 new_mask 为 True 的 patch tokens。
    new_mask: [B, patch_h, patch_w]，会先 flatten 成 [B, num_patches]。
    """
    B, patch_h, patch_w = new_mask.shape
    num_patches = patch_h * patch_w
    keep_flat = new_mask.reshape(B, num_patches)
    # 索引：前 patch_start_idx 个全保留，后面只保留 keep_flat 为 True 的
    special_indices = torch.arange(patch_start_idx, device=new_mask.device)
    patch_indices = torch.where(keep_flat[0])[0] + patch_start_idx
    if B > 1:
        raise NotImplementedError("filter_kv_list_by_patch_mask only supports B=1")
    keep_indices = torch.cat([special_indices, patch_indices], dim=0)

    filtered_list = []
    for (k, v) in new_kv_list:
        # k, v: [B, num_heads, seq_len, head_dim]
        k_f = k[:, :, keep_indices, :]
        v_f = v[:, :, keep_indices, :]
        filtered_list.append((k_f, v_f))
    return filtered_list


@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None

class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, total_budget=1200000):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.total_budget = total_budget
    


    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0
    ):
        images = torch.stack(
            [view["img"] for view in views], dim=0
        ).permute(1, 0, 2, 3, 4)    # B S C H W

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf
            predictions["images"] = images

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
                    'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

                    'depth': predictions['depth'][:, s],  # [B, H, W, 1]
                    'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
                    'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]

                    **({'valid_mask': views[s]["valid_mask"]}
                    if 'valid_mask' in views[s] else {}),  # [B, H, W]

                    **({'track': predictions['track'][:, s],  # [B, N, 2]
                        'vis': predictions['vis'][:, s],  # [B, N]
                        'track_conf': predictions['conf'][:, s]}
                    if 'track' in predictions else {})
                }
                ress.append(res)
            return StreamVGGTOutput(ress=ress, views=views)  # [S] [B, C, H, W]
    
    def inference(
        self,
        frames,
        query_points=None,
        past_key_values=None,
        frame_writer=None,
        cache_results: bool = True,
        total_budget=None,
        filter_new_region: bool = True,
        new_region_dist_threshold: float = 0.02,
        new_region_conf_threshold: float = 0.3,
        max_seen_points: int = 100000000,
        # ── 回环参数 ─────────────────────────────────────────────────────────
        loop_closure_first_k: int = 5,
        loop_closure_last_k: int = 1,
        loop_closure_neighbor_k: int = 21,
        loop_closure_outlier_pct: float = 10.0,
        use_voxel_seen_points_v2: bool = False,
        max_frames: int = 20,
    ):
        from .loop_closure_kv import (
            FrameBBoxBank,
            select_relevant_frame_indices,
            filter_past_kv_by_frame_indices,
        )

        # ── 原有初始化 ────────────────────────────────────────────────────────
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        total_budget = self.total_budget
        seen_points_3d = None
        patch_size = self.aggregator.patch_size
        patch_start_idx = self.aggregator.patch_start_idx

        all_ress = []
        processed_frames = []
        kv_img_idx = [[] for _ in range(self.aggregator.depth)]
        kv_img_idx_camera = [[] for _ in range(self.camera_head.trunk_depth)]

        # ── [NEW] 回环相关初始化 ──────────────────────────────────────────────
        bbox_bank = FrameBBoxBank(outlier_percentile=loop_closure_outlier_pct)
        all_processed_frame_indices: List[int] = []
        _last_pts3d_patch: Optional[torch.Tensor] = None
        # _last_new_mask: Optional[torch.Tensor] = None

                # ── [NEW] 最近 N 帧完整 KV cache ─────────────────────────────────────
        RECENT_FULL_FRAMES = 0
        # 每个元素: (frame_idx, pruned_kv_list_cpu)
        # 逻辑：cache 先写 full token，超过窗口后再替换为该帧的 pruned token。
        recent_full_kv_buffer = []   # 最多保留 RECENT_FULL_FRAMES 条


        from tqdm import tqdm
        # ── 主循环 ────────────────────────────────────────────────────────────
        for i, frame in tqdm(enumerate(frames)):
            images = frame["img"].unsqueeze(0)

            # ── [NEW] 用上一帧 bbox 近似当前帧，选出相关历史 KV ──────────────
            if i > 0 and _last_pts3d_patch is not None:
                relevant_indices = select_relevant_frame_indices(
                    current_frame_idx=i,
                    all_processed_frame_indices=all_processed_frame_indices,
                    bbox_bank=bbox_bank,
                    current_pts3d_patch=_last_pts3d_patch,
                    # current_new_mask=_last_new_mask,
                    first_k=loop_closure_first_k,
                    last_k=loop_closure_last_k,
                    neighbor_k=loop_closure_neighbor_k,
                )
                # print(relevant_indices)
                agg_past_kv, _ = filter_past_kv_by_frame_indices(
                    past_key_values, kv_img_idx, relevant_indices, target_device=images.device
                )
                print(i, relevant_indices)
            else:
                # 未做帧选择时，仍需确保传入 aggregator 的 KV 在当前 images 的 device 上
                agg_past_kv = move_kv_cache_to_device(past_key_values, images.device)

            # ── aggregator 推理 ───────────────────────────────────────────────
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
                if self.camera_head is not None:
                    pose_enc, block_kv_list = self.camera_head(
                        aggregated_tokens,
                        past_key_values_camera=past_key_values_camera,
                        use_cache=True,
                    )
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    depth = depth[:, 0]
                    depth_conf = depth_conf[:, 0]

                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    pts3d = pts3d[:, 0]
                    pts3d_conf = pts3d_conf[:, 0]

                # ── 新见区域过滤 + [NEW] 存当前帧 bbox ───────────────────────
                if filter_new_region and self.point_head is not None:
                    H, W = images.shape[-2], images.shape[-1]
                    ph, pw = H // patch_size, W // patch_size
                    new_mask, pts3d_patch = compute_new_region_mask(
                        pts3d, pts3d_conf, seen_points_3d,
                        patch_size, ph, pw,
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
                            if use_voxel_seen_points_v2:
                                seen_points_3d = VoxelSeenPoints_v2(
                                    voxel_size=new_region_dist_threshold,
                                    device=new_pts.device,
                                    max_voxels=max_seen_points,
                                    max_frames=max_frames,
                                )
                            else:
                                seen_points_3d = VoxelSeenPoints(
                                    voxel_size=new_region_dist_threshold,
                                    device=new_pts.device,
                                    max_voxels=max_seen_points,
                                    max_frames=max_frames,
                                )
                        seen_points_3d.add_points(new_pts, frame_idx=i)

                    # [NEW] 存 bbox，留给下一帧用
                    bbox_bank.add_frame(i, pts3d_patch, new_mask)
                    _last_pts3d_patch = pts3d_patch.detach()
                    # _last_new_mask = new_mask.detach()

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images,
                        patch_start_idx=patch_start_idx, query_points=query_points,
                    )
                    track = track_list[-1][:, 0]
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            # ── 完整 KV cache 照常更新（剪枝只影响 aggregator 输入，不影响存储）──
            past_key_values, kv_img_idx = update_kv_cache(
                past_key_values, new_kv_list_to_cache, kv_img_idx, i
            )
            # 将 aggregator 的 KV cache 统一移到 CPU，仅在下次被选中使用时再搬到 CUDA
            # past_key_values = move_kv_cache_to_device(past_key_values, torch.device("cpu"))

                        # 记录该帧 pruned token（放 CPU 减少显存占用）
            pruned_kv_list_cpu = [
                (k.detach().cpu(), v.detach().cpu()) for (k, v) in new_kv_list_to_cache
            ]
            recent_full_kv_buffer.append((i, pruned_kv_list_cpu))

            # 超过最近窗口后，把最旧帧从 full 替换为 pruned
            if len(recent_full_kv_buffer) > RECENT_FULL_FRAMES:
                old_frame_idx, old_pruned_kv = recent_full_kv_buffer.pop(0)
                past_key_values, kv_img_idx = replace_frame_kv_in_cache(
                    past_key_values, kv_img_idx, old_frame_idx, old_pruned_kv
                )



            past_key_values_camera, kv_img_idx_camera = update_kv_cache(
                past_key_values_camera, block_kv_list, kv_img_idx_camera, i
            )

            # [NEW] 记录已处理帧
            all_processed_frame_indices.append(i)

            # ── 结果收集 ──────────────────────────────────────────────────────
            res_gpu = {
                "pts3d_in_other_view": pts3d,
                "conf": pts3d_conf,
                "depth": depth,
                "depth_conf": depth_conf,
                "camera_pose": camera_pose,
                **({"valid_mask": frame["valid_mask"]} if "valid_mask" in frame else {}),
                **(
                    {"track": track, "vis": vis, "track_conf": track_conf}
                    if query_points is not None else {}
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
                    {nk: nv.detach().cpu() if isinstance(nv, torch.Tensor) else nv
                     for nk, nv in frame.items()}
                )
            del res_gpu
            torch.cuda.empty_cache()

        return StreamVGGTOutput(
            ress=all_ress if cache_results else None,
            views=processed_frames if cache_results else None,
        )
