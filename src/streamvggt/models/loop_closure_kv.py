"""
loop_closure_kv.py
==================
类 SLAM 回环检测的 KV cache 剪枝策略。

对每一帧推理，只保留 past_key_values 中与当前区域"有关"的 token：
    1. 前 FIRST_K 张图像的全部 token
    2. 最近 LAST_K 张图像的全部 token
    3. AABB 重叠检测：记录每帧 3D 点的去极端值后的 bbox，
       与当前帧 bbox 有交集的帧及其 ±NEIGHBOR 帧的全部 token

核心数据结构：
    FrameBBoxBank：为每帧存储一个 3D AABB（min_xyz, max_xyz），
                   判断重叠只需 6 次比较，复杂度 O(N_frames)。
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List, Optional, Set, Tuple, Dict


# ────────────────────────────────────────────────────────────────────────────────
# 1. FrameBBoxBank  —  按帧存储去极端值后的 3D AABB
# ────────────────────────────────────────────────────────────────────────────────

class FrameBBoxBank:
    """
    为每一帧维护一个 3D 轴对齐包围盒（AABB）。

    add_frame 时：
        1. 去除每轴 outlier_percentile% 之外的极端点
        2. 对剩余点取 min/max，得到该帧的 bbox

    find_overlapping_frames 时：
        对所有历史帧做 AABB 交集测试（O(N_frames)，每帧只需 6 次比较）。

    outlier_percentile: 去极端值的百分位，默认 2.0（去掉最小/最大各 2%）
    """

    def __init__(self, outlier_percentile: float = 2.0):
        self.outlier_percentile = outlier_percentile
        # frame_idx -> (min_xyz, max_xyz)，均为 shape (3,) 的 CPU float tensor
        self._bboxes: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ── 工具 ────────────────────────────────────────────────────────────────────

    def _pts_to_bbox(
        self, pts: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        去除极端值后计算 bbox。
        pts: (N, 3) float，任意 device。
        返回 (min_xyz, max_xyz) 均在 CPU，或 None（点不足时）。
        """
        pts = pts.float().cpu()
        if pts.shape[0] < 4:
            return None

        p = self.outlier_percentile / 100.0
        if p > 0:
            lo = torch.quantile(pts, p, dim=0)        # (3,)
            hi = torch.quantile(pts, 1.0 - p, dim=0)  # (3,)
            mask = ((pts >= lo) & (pts <= hi)).all(dim=1)
            pts = pts[mask]

        if pts.shape[0] == 0:
            return None

        return pts.min(dim=0).values, pts.max(dim=0).values

    @staticmethod
    def _aabb_intersects(
        min_a: torch.Tensor, max_a: torch.Tensor,
        min_b: torch.Tensor, max_b: torch.Tensor,
    ) -> bool:
        """
        判断两个 3D AABB 是否有“足够大”的交集：
            1. 先判断是否有几何交集（含边界接触）；
            2. 再要求交集体积至少占当前帧（min_a, max_a）体积的 10%。
        """
        # 先做快速几何相交测试
        has_intersection = (
            (min_a[0] <= max_b[0]) and (max_a[0] >= min_b[0]) and
            (min_a[1] <= max_b[1]) and (max_a[1] >= min_b[1]) and
            (min_a[2] <= max_b[2]) and (max_a[2] >= min_b[2])
        )
        if not bool(has_intersection):
            return False

        # 计算交集 AABB
        inter_min = torch.max(min_a, min_b)
        inter_max = torch.min(max_a, max_b)
        inter_sizes = (inter_max - inter_min).clamp(min=0.0)
        inter_vol = float(inter_sizes.prod().item())

        # 当前帧（A）的体积
        a_sizes = (max_a - min_a).clamp(min=0.0)
        a_vol = float(a_sizes.prod().item())
        if a_vol <= 0.0:
            # 退化 bbox，保守返回 False
            return False

        overlap_ratio = inter_vol / a_vol
        return overlap_ratio >= 0.01

    # ── 公开 API ─────────────────────────────────────────────────────────────────

    def add_frame(
        self,
        frame_idx: int,
        pts3d_patch: torch.Tensor,   # [B, ph, pw, 3]
        new_mask: torch.Tensor,      # [B, ph, pw] bool
    ) -> None:
        """
        计算本帧新见区域 3D 点的 bbox 并存入 bank。仅支持 B=1。
        """
        pts = pts3d_patch[0].reshape(-1, 3)  # (ph*pw, 3)
        mask = new_mask[0].reshape(-1)        # (ph*pw,)
        pts = pts[mask].detach()

        result = self._pts_to_bbox(pts)
        if result is not None:
            self._bboxes[frame_idx] = result

    def get_bbox(self, frame_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self._bboxes.get(frame_idx, None)

    def frame_indices(self) -> List[int]:
        return sorted(self._bboxes.keys())

    # ── AABB 重叠检测 ─────────────────────────────────────────────────────────────

    def find_overlapping_frames(
        self,
        query_pts: torch.Tensor,   # (Q, 3)  当前帧 3D 点（内部自动去极端值）
    ) -> List[int]:
        """
        返回与当前帧 bbox 有交集的所有历史帧编号。
        复杂度 O(N_frames)，每帧仅 6 次标量比较。
        """
        result = self._pts_to_bbox(query_pts)
        if result is None:
            return []
        cur_min, cur_max = result
        # for fidx, (fmin, fmax) in self._bboxes.items():
        #     print(fidx, fmin, fmax)

        return [
            fidx for fidx, (fmin, fmax) in self._bboxes.items()
            if self._aabb_intersects(cur_min, cur_max, fmin, fmax)
        ]


# ────────────────────────────────────────────────────────────────────────────────
# 2. select_relevant_frame_indices  —  综合三条规则，返回应保留的帧编号集合
# ────────────────────────────────────────────────────────────────────────────────

def select_relevant_frame_indices(
    current_frame_idx: int,
    all_processed_frame_indices: List[int],
    bbox_bank: FrameBBoxBank,
    current_pts3d_patch: torch.Tensor,   # [B, ph, pw, 3]  上一帧近似
    # current_new_mask: torch.Tensor,      # [B, ph, pw] bool
    first_k: int = 10,
    last_k: int = 10,
    neighbor_k: int = 5,
) -> Set[int]:
    """
    返回本轮 aggregator 推理应使用的历史帧编号集合。

    规则：
        1. 前 first_k 帧
        2. 最近 last_k 帧（不含当前帧）
        3. 与当前帧 bbox 有交集的历史帧 ± neighbor_k 帧
    """
    all_prev = [f for f in all_processed_frame_indices if f < current_frame_idx]
    if not all_prev:
        return set()

    keep: Set[int] = set()
    valid_set = set(all_prev)

    # ── 规则 1：前 first_k 帧 ─────────────────────────────────────────────────
    keep.update(all_prev[:first_k])

    # ── 规则 2：最近 last_k 帧 ────────────────────────────────────────────────
    keep.update(all_prev[-last_k:])

    # # ── 规则 3：bbox 有交集的帧 ± neighbor_k ─────────────────────────────────
    pts = current_pts3d_patch[0].reshape(-1, 3)
    query_pts = pts.detach()
    relative_frames = set()
    if current_frame_idx == 81:
        pass
    if query_pts.shape[0] > 0:
        for rf in bbox_bank.find_overlapping_frames(query_pts):
            for nb in range(rf - neighbor_k, rf + neighbor_k + 1):
                if nb in valid_set:
                    keep.add(nb)
                    relative_frames.add(nb)
    # print(current_frame_idx, ":", keep & valid_set)
    return keep & valid_set


# ────────────────────────────────────────────────────────────────────────────────
# 3. filter_past_kv_by_frame_indices  —  从 past_key_values 中取出指定帧的 token
# ────────────────────────────────────────────────────────────────────────────────

def filter_past_kv_by_frame_indices(
    past_key_values: List,
    kv_img_idx: List[List[int]],
    keep_frame_indices: Set[int],
    target_device: Optional[torch.device] = None,
) -> Tuple[List, List[List[int]]]:
    """
    从 past_key_values 中取出属于 keep_frame_indices 的 token。

    past_key_values: List[ (k, v) ]，k/v shape = [B, heads, seq, dim]
    kv_img_idx:      List[ List[int] ]，每层每个 token 对应的帧编号
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
        if (target_device is not None) and (k_sel.device != target_device):
            k_sel = k_sel.to(target_device)
            v_sel = v_sel.to(target_device)
        filtered_past_kv.append((k_sel, v_sel))
        filtered_kv_idx.append(idx_arr[keep_positions].tolist())

    return filtered_past_kv, filtered_kv_idx


# ────────────────────────────────────────────────────────────────────────────────
# 4. build_patched_inference  —  替换 StreamVGGT.inference 的完整方法
# ────────────────────────────────────────────────────────────────────────────────

def build_patched_inference():
    """
    返回可直接替换 StreamVGGT.inference 的新方法。

    使用方式：
        from loop_closure_kv import build_patched_inference
        StreamVGGT.inference = build_patched_inference()
    """

    def inference(
        self,
        frames,
        query_points=None,
        past_key_values=None,
        frame_writer=None,
        cache_results: bool = True,
        total_budget=None,
        filter_new_region: bool = True,
        new_region_dist_threshold: float = 0.05,
        new_region_conf_threshold: float = 0.3,
        max_seen_points: int = 500000,
        # ── 回环参数 ─────────────────────────────────────────────────────────
        loop_closure_first_k: int = 10,
        loop_closure_last_k: int = 10,
        loop_closure_neighbor_k: int = 5,
        loop_closure_outlier_pct: float = 2.0,
    ):
        from loop_closure_kv import (
            FrameBBoxBank,
            select_relevant_frame_indices,
            filter_past_kv_by_frame_indices,
        )
        from streamvggt.models.model import (
            update_kv_cache,
            compute_new_region_mask,
            filter_kv_list_by_patch_mask,
            VoxelSeenPoints,
            StreamVGGTOutput,
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
        _last_new_mask: Optional[torch.Tensor] = None

        # ── 主循环 ────────────────────────────────────────────────────────────
        for i, frame in enumerate(frames):
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
                agg_past_kv, _ = filter_past_kv_by_frame_indices(
                    past_key_values, kv_img_idx, relevant_indices
                )
            else:
                agg_past_kv = past_key_values

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
                            seen_points_3d = VoxelSeenPoints(
                                voxel_size=new_region_dist_threshold,
                                device=new_pts.device,
                                max_voxels=max_seen_points,
                                max_frames=30,
                            )
                        seen_points_3d.add_points(new_pts, frame_idx=i)

                    # [NEW] 存 bbox，留给下一帧用
                    bbox_bank.add_frame(i, pts3d_patch, new_mask)
                    _last_pts3d_patch = pts3d_patch.detach()
                    _last_new_mask = new_mask.detach()

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

    return inference


# ────────────────────────────────────────────────────────────────────────────────
# smoke test
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== FrameBBoxBank smoke test ===")
    bank = FrameBBoxBank(outlier_percentile=2.0)

    # 帧 0-2：原点附近
    for fi in range(3):
        pts = torch.randn(200, 3) * 0.3
        patch = pts.reshape(1, 10, 20, 3)
        mask = torch.ones(1, 10, 20, dtype=torch.bool)
        bank.add_frame(fi, patch, mask)
        bb = bank.get_bbox(fi)
        print(f"  frame {fi} bbox: {bb[0].numpy().round(2)} ~ {bb[1].numpy().round(2)}")

    # 帧 3-4：远处（x+10）
    for fi in range(3, 5):
        pts = torch.randn(200, 3) * 0.3 + 10.0
        patch = pts.reshape(1, 10, 20, 3)
        mask = torch.ones(1, 10, 20, dtype=torch.bool)
        bank.add_frame(fi, patch, mask)

    # 查询：靠近原点 → 应命中帧 0,1,2
    query = torch.randn(100, 3) * 0.3
    print(f"\n  query near origin → overlapping frames (expect 0,1,2): "
          f"{sorted(bank.find_overlapping_frames(query))}")

    # 查询：靠近远处 → 应命中帧 3,4
    query_far = torch.randn(100, 3) * 0.3 + 10.0
    print(f"  query near far region → overlapping frames (expect 3,4): "
          f"{sorted(bank.find_overlapping_frames(query_far))}")

    print("\n=== select_relevant_frame_indices smoke test ===")
    cur_patch = torch.randn(1, 10, 20, 3) * 0.3
    cur_mask = torch.ones(1, 10, 20, dtype=torch.bool)
    keep = select_relevant_frame_indices(
        current_frame_idx=20,
        all_processed_frame_indices=list(range(20)),
        bbox_bank=bank,
        current_pts3d_patch=cur_patch,
        current_new_mask=cur_mask,
        first_k=3, last_k=3, neighbor_k=2,
    )
    print(f"  kept frames: {sorted(keep)}")
    print("  (expect: 0,1,2 [first3] + 17,18,19 [last3] + bbox_overlap_frames±2)")
