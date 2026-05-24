from __future__ import annotations

from typing import List, Tuple

import torch


def replace_frame_kv_in_cache(
    past_key_values: List,
    kv_img_idx: List,
    frame_idx: int,
    replacement_kv_list: List,
) -> Tuple[List, List]:
    """Replace the KV tokens for a specific frame with a new (typically pruned) set.

    Tokens belonging to the same frame are assumed to be contiguous in the
    sequence dimension (guaranteed by the append logic in ``update_kv_cache``).

    Args:
        past_key_values: Per-layer list of ``(k, v)`` tensors,
            each with shape ``[B, heads, seq, dim]``.
        kv_img_idx: Per-layer list of frame-index labels,
            parallel to the sequence dimension.
        frame_idx: The frame whose tokens should be replaced.
        replacement_kv_list: New ``(k, v)`` pairs for each layer.

    Returns:
        Updated ``(past_key_values, kv_img_idx)`` tuple.
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


def update_kv_cache(
    past_key_values: List,
    new_kv_list: List,
    kv_img_idx: List,
    img_idx: int,
) -> Tuple[List, List]:
    """Append new KV pairs to the existing cache.

    If a layer's cache slot is ``None`` it is initialised directly;
    otherwise the new tensors are concatenated along the sequence dimension.

    Args:
        past_key_values: Existing per-layer ``(k, v)`` cache.
        new_kv_list: New ``(k, v)`` pairs produced for the current frame.
        kv_img_idx: Per-layer sequence of frame-index labels.
        img_idx: Frame index to assign to the newly appended tokens.

    Returns:
        Updated ``(past_key_values, kv_img_idx)`` tuple.
    """
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


def move_kv_cache_to_device(
    past_key_values: List,
    device: torch.device,
) -> List:
    """Move all KV tensors in the cache to the specified device in-place.

    Args:
        past_key_values: Per-layer ``(k, v)`` cache (may contain ``None`` slots).
        device: Target device.

    Returns:
        The same list with tensors relocated to ``device``.
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
