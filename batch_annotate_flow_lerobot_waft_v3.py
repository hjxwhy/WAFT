"""
Optimized batch annotation of LeRobot v3.0 datasets with WAFT optical flow.

3-stage async pipeline for maximum throughput:
  Stage 1 - Prefetch:  Multi-process video decoding (ProcessPoolExecutor)
  Stage 2 - Inference: Batched GPU calc_flow (main thread)
  Stage 3 - Postprocess: Multi-process mask generation + RLE save (ProcessPoolExecutor)

The GPU never waits for video decoding or CPU post-processing.

Unchanged from v1 (batch_annotate_flow_lerobot_waft.py):
  - Output format: per-episode per-camera JSON with pycocotools RLE masks
  - compensate_ego_motion, generate_robust_motion_mask, morphology
  - ROI cropping, resize-to-640, all mask thresholds
  - save_episode_visualization_sparse

Usage:
    python batch_annotate_flow_lerobot_waft_v3.py \\
        --data_root /path/to/datasets/ \\
        --output_root /path/to/output \\
        --cfg config/a2/twins/tar-c-t.json \\
        --ckpt path/to/checkpoint.pth \\
        --num_prefetch_workers 4 --num_postprocess_workers 4 \\
        --max_flow_batch 16 --prefetch_buffer 8 --fp16
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import pycocotools.mask as mask_util

# ---------------------------------------------------------------------------
# Path setup (same as v1)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.parser import parse_args as waft_parse_args
from model import fetch_model
from utils.utils import load_ckpt
from inference_tools import InferenceWrapper

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_repo_root, "UFM"))
from generate_mask import generate_robust_motion_mask
from lerobot.datasets.video_utils import decode_video_frames_torchvision

# ---------------------------------------------------------------------------
# Camera maps (unchanged from v1)
# ---------------------------------------------------------------------------
VALID_CAMERAS = {
    'arx5': ['cam_high', 'cam_side'],
    'ur5': ['cam_high'],
    'franka': ['cam_high', 'cam_side'],
    'aloha': ['cam_high'],
    'r1lite': ['observation.images.head_rgb'],
    "unitree_g1": ["observation.images.head_stereo_left",
                    "observation.images.head_stereo_right"],
    "widowx": ["observation.images.image_0", "observation.images.image_1",
                "observation.images.image_2", "observation.images.image_3"],
    "google_robot": ["observation.images.image"],
}

TARGET_CAMERAS = [
    'cam_high', 'cam_side', 'observation.images.head_rgb',
    'observation.images.head_stereo_left',
    "observation.images.image_0", "observation.images.image_1",
    "observation.images.image_2", "observation.images.image_3",
    "observation.images.image",
]


# ===================================================================
# Functions preserved verbatim from v1
# ===================================================================

def load_dataset_info(dataset_path):
    dataset_path = Path(dataset_path)
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    robot_type = info.get('robot_type', 'unknown')
    fps = info.get('fps', 30)
    valid_cams = VALID_CAMERAS.get(robot_type, ['cam_high'])
    active_cameras = [cam for cam in valid_cams if cam in TARGET_CAMERAS]
    episodes = []
    episodes_dir = dataset_path / "meta" / "episodes"
    for parquet_file in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(str(parquet_file))
        episodes.extend(df.to_dict('records'))
    return {
        'info': info,
        'robot_type': robot_type,
        'active_cameras': active_cameras,
        'episodes': episodes,
        'fps': fps,
        'video_path_template': info.get(
            'video_path',
            'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'),
    }


def compensate_ego_motion(flow, ransac_reproj_threshold=3.0):
    """
    Remove background ego-motion from optical flow using Homography RANSAC.
    """
    h, w = flow.shape[1:]
    step = max(1, int(math.sqrt(h * w / 10000)))
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    src = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    dst = src + np.stack([
        flow[0, ::step, ::step].ravel(),
        flow[1, ::step, ::step].ravel()
    ], axis=-1).astype(np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_reproj_threshold)
    if H is None:
        return flow
    xx_grid, yy_grid = np.meshgrid(
        np.arange(w, dtype=np.float64),
        np.arange(h, dtype=np.float64))
    denom = H[2, 0] * xx_grid + H[2, 1] * yy_grid + H[2, 2]
    pred_x = (H[0, 0] * xx_grid + H[0, 1] * yy_grid + H[0, 2]) / denom
    pred_y = (H[1, 0] * xx_grid + H[1, 1] * yy_grid + H[1, 2]) / denom
    pred_flow = np.zeros_like(flow)
    pred_flow[0] = (pred_x - xx_grid).astype(np.float32)
    pred_flow[1] = (pred_y - yy_grid).astype(np.float32)
    return flow - pred_flow


def compute_needed_frame_indices(total_frames, fps, max_fps, time_offset,
                                 short_time_offset):
    if max_fps is not None and max_fps > 0 and fps > max_fps:
        downsample_factor = int(math.ceil(fps / max_fps))
    else:
        downsample_factor = 1
    effective_fps = fps / downsample_factor
    offset_frames = int(time_offset * effective_fps)
    offset_frames_short = max(1, int(short_time_offset * effective_fps))
    sampled_indices = list(range(0, total_frames, downsample_factor))
    num_valid = len(sampled_indices) - offset_frames
    needed = set()
    for i in range(max(0, num_valid)):
        needed.add(sampled_indices[i])
        needed.add(sampled_indices[i + offset_frames])
        short_j = max(0, i + offset_frames - offset_frames_short)
        needed.add(sampled_indices[short_j])
    return sorted(needed)


def save_episode_optical_flow(masks, output_path, episode_info):
    if not masks:
        return None
    rle_data = {
        'episode_index': episode_info['episode_index'],
        'dataset': episode_info['dataset'],
        'camera_key': episode_info['camera_key'],
        'robot_type': episode_info['robot_type'],
        'num_frames': len(masks),
        'time_offset': episode_info['time_offset'],
        'fps': episode_info['fps'],
        'metadata': {
            'image_size': list(list(masks.values())[0].shape),
            'from_timestamp': episode_info['from_timestamp'],
            'to_timestamp': episode_info['to_timestamp'],
            'video_file': episode_info['video_file'],
            'episode_length': episode_info['episode_length'],
            'rle_format': 'pycocotools',
        },
        'frames': []
    }
    for frame_idx, mask in sorted(masks.items()):
        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')
        rle['frame_idx'] = frame_idx
        rle_data['frames'].append(rle)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(rle_data, f, indent=2)
    original_size = sum(m.nbytes for m in masks.values())
    compressed_size = output_path.stat().st_size
    compression_ratio = (original_size / compressed_size
                         if compressed_size > 0 else 1.0)
    return {
        'original_size_mb': original_size / 1024 / 1024,
        'compressed_size_mb': compressed_size / 1024 / 1024,
        'compression_ratio': compression_ratio,
        'num_frames': len(masks),
    }


def save_episode_visualization_sparse(
    frames_dict, masks, flows_dict, output_dir, camera_key,
    episode_idx, time_offset_frames=30, max_vis_frames=10,
):
    import flow_vis as _flow_vis

    vis_dir = output_dir / "visualizations" / \
        f"episode_{episode_idx:06d}_{camera_key}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    sorted_keys = sorted(masks.keys())[:max_vis_frames]
    for frame_idx in sorted_keys:
        source = frames_dict.get(frame_idx)
        if source is None:
            continue
        target_idx = frame_idx + time_offset_frames
        target = frames_dict.get(target_idx, source)
        mask = masks[frame_idx]
        flow, covis = None, None
        if flows_dict and frame_idx in flows_dict:
            flow, covis = flows_dict[frame_idx]
        vis_images = [source, target]
        overlay = source.copy().astype(np.float32)
        mask_3ch = np.stack([mask, mask, mask], axis=-1).astype(np.float32)
        overlay = (overlay * (1 - mask_3ch * 0.5)
                   + np.array([255, 0, 0], dtype=np.float32) * mask_3ch * 0.5)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        vis_images.append(overlay)
        if flow is not None:
            flow_vis_img = _flow_vis.flow_to_color(flow.transpose(1, 2, 0))
            vis_images.append(flow_vis_img)
            flow_magnitude = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
            magnitude_normalized = (
                flow_magnitude / (flow_magnitude.max() + 1e-6) * 255
            ).astype(np.uint8)
            magnitude_colormap = cv2.applyColorMap(
                magnitude_normalized, cv2.COLORMAP_JET)
            magnitude_colormap = cv2.cvtColor(
                magnitude_colormap, cv2.COLOR_BGR2RGB)
            vis_images.append(magnitude_colormap)
        target_h = source.shape[0]
        vis_resized = []
        for img in vis_images:
            h, w = img.shape[:2]
            if h != target_h:
                new_w = int(w * target_h / h)
                img = cv2.resize(img, (new_w, target_h))
            vis_resized.append(img)
        combined = np.concatenate(vis_resized, axis=1)
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(vis_dir / f"frame_{frame_idx:06d}.jpg"),
            combined_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])


# ===================================================================
# Stage 1: Prefetch -- runs in subprocess
# ===================================================================

def prefetch_episode(task):
    """
    Decode video frames and compute triplet indices for one
    (episode, camera) pair.  Runs in a subprocess via ProcessPoolExecutor.

    Args:
        task: dict with keys dataset_path, episode (dict), camera_key,
              fps, time_offset, short_time_offset, max_fps, skip_existing,
              output_root, dataset_name, robot_type, use_bidirectional.

    Returns:
        dict with decoded frames, triplets, and metadata -- or None if
        the episode should be skipped.
    """
    dataset_path = Path(task['dataset_path'])
    episode = task['episode']
    cam_key = task['camera_key']
    fps = task['fps']
    ep_idx = episode['episode_index']
    episode_length = episode['length']

    output_path = (Path(task['output_root']) / task['dataset_name']
                   / cam_key / f"episode_{ep_idx:06d}.json")

    if task['skip_existing'] and output_path.exists():
        return None

    from_ts = episode[f'videos/{cam_key}/from_timestamp']
    to_ts = episode[f'videos/{cam_key}/to_timestamp']
    total_frames = int((to_ts - from_ts) * fps)

    needed_indices = compute_needed_frame_indices(
        total_frames, fps, task['max_fps'],
        task['time_offset'], task['short_time_offset'],
    )
    if not needed_indices:
        return None

    chunk_idx = episode[f'videos/{cam_key}/chunk_index']
    file_idx = episode[f'videos/{cam_key}/file_index']
    video_path = (dataset_path / "videos" / cam_key
                  / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4")
    if not video_path.exists():
        return None

    timestamps = [float(from_ts + i / fps) for i in needed_indices]
    try:
        frames_tensor = decode_video_frames_torchvision(
            video_path=video_path,
            timestamps=timestamps,
            tolerance_s=1.0 / fps,
            backend="pyav",
            log_loaded_timestamps=False,
        )
    except Exception:
        return None

    frames_np = frames_tensor.permute(0, 2, 3, 1).numpy()
    if frames_np.dtype == np.float32 and frames_np.max() <= 1.0:
        frames_np = (frames_np * 255).astype(np.uint8)
    elif frames_np.dtype != np.uint8:
        frames_np = frames_np.astype(np.uint8)

    frame_dict = {}
    for i, fi in enumerate(needed_indices):
        if i < len(frames_np):
            frame_dict[fi] = frames_np[i]

    if not frame_dict:
        return None

    first_frame = frame_dict[needed_indices[0]]
    if np.all(first_frame < 1):
        return None

    # Compute triplets (src_idx, tgt_long_idx, tgt_short_idx)
    if task['max_fps'] is not None and task['max_fps'] > 0 and fps > task['max_fps']:
        downsample_factor = int(math.ceil(fps / task['max_fps']))
    else:
        downsample_factor = 1
    effective_fps = fps / downsample_factor
    offset_frames = int(task['time_offset'] * effective_fps)
    offset_frames_short = max(1, int(task['short_time_offset'] * effective_fps))
    sampled_indices = list(range(0, total_frames, downsample_factor))
    num_valid = len(sampled_indices) - offset_frames

    triplets = []
    for i in range(max(0, num_valid)):
        src_idx = sampled_indices[i]
        tgt_long_idx = sampled_indices[i + offset_frames]
        short_j = max(0, i + offset_frames - offset_frames_short)
        tgt_short_idx = sampled_indices[short_j]
        triplets.append((src_idx, tgt_long_idx, tgt_short_idx))

    if not triplets:
        return None

    return {
        'episode_index': ep_idx,
        'camera_key': cam_key,
        'frames': frame_dict,
        'triplets': triplets,
        'total_frames': total_frames,
        'episode_length': episode_length,
        'from_timestamp': from_ts,
        'to_timestamp': to_ts,
        'video_file': f"chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4",
        'output_path': str(output_path),
        'dataset_name': task['dataset_name'],
        'robot_type': task['robot_type'],
        'fps': fps,
    }


# ===================================================================
# Stage 2: Batched GPU inference -- runs on main thread
# ===================================================================

@torch.no_grad()
def build_pairs_and_infer(wrapped_model, episode_data, device, roi,
                          max_flow_batch, use_bidirectional, use_fp16,
                          save_vis):
    """
    Collect all (img1, img2) pairs from one episode, tag each with
    (pair_type, tgt_long_idx), run batched calc_flow, and return
    flows grouped by tgt_long_idx.

    Returns:
        flows_by_frame: {tgt_long_idx: {"long": ndarray(2,H,W), ...}}
        geometry: dict with roi bounds, scale info, original size
    """
    frames = episode_data['frames']
    triplets = episode_data['triplets']

    sample_frame = next(iter(frames.values()))
    original_h, original_w = sample_frame.shape[:2]

    if roi is not None:
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(original_w, x2), min(original_h, y2)
    else:
        x1, y1, x2, y2 = 0, 0, original_w, original_h
    roi_w, roi_h = x2 - x1, y2 - y1
    roi_tuple = (x1, y1, x2, y2)

    longest_edge = max(roi_h, roi_w)
    if longest_edge > 640:
        scale_factor = 640 / longest_edge
        new_w = int(roi_w * scale_factor)
        new_h = int(roi_h * scale_factor)
        scale_info = (new_w, new_h)
    else:
        scale_factor = None
        scale_info = None

    geometry = {
        'original_h': original_h, 'original_w': original_w,
        'roi': roi_tuple, 'roi_w': roi_w, 'roi_h': roi_h,
        'scale_factor': scale_factor, 'scale_info': scale_info,
    }

    # --- Collect all pairs with metadata ---
    pairs_img1 = []
    pairs_img2 = []
    pairs_info = []  # (pair_type, tgt_long_idx)

    for src_idx, tgt_long_idx, tgt_short_idx in triplets:
        if src_idx not in frames or tgt_long_idx not in frames:
            continue

        src_roi = frames[src_idx][y1:y2, x1:x2]
        tgt_long_roi = frames[tgt_long_idx][y1:y2, x1:x2]

        if scale_info is not None:
            src_roi = cv2.resize(src_roi, scale_info,
                                 interpolation=cv2.INTER_LINEAR)
            tgt_long_roi = cv2.resize(tgt_long_roi, scale_info,
                                      interpolation=cv2.INTER_LINEAR)

        # Long backward flow: tgt_long -> src
        pairs_img1.append(tgt_long_roi)
        pairs_img2.append(src_roi)
        pairs_info.append(("long", tgt_long_idx))

        if tgt_short_idx != tgt_long_idx and tgt_short_idx in frames:
            tgt_short_roi = frames[tgt_short_idx][y1:y2, x1:x2]
            if scale_info is not None:
                tgt_short_roi = cv2.resize(tgt_short_roi, scale_info,
                                           interpolation=cv2.INTER_LINEAR)
            # Short forward: tgt_long -> tgt_short
            pairs_img1.append(tgt_long_roi)
            pairs_img2.append(tgt_short_roi)
            pairs_info.append(("short_fwd", tgt_long_idx))

            # Short backward: tgt_short -> tgt_long
            if use_bidirectional:
                pairs_img1.append(tgt_short_roi)
                pairs_img2.append(tgt_long_roi)
                pairs_info.append(("short_bwd", tgt_long_idx))

    if not pairs_img1:
        return {}, geometry, None

    # --- Batched inference ---
    all_flows = []
    for start in range(0, len(pairs_img1), max_flow_batch):
        end = min(start + max_flow_batch, len(pairs_img1))
        batch_img1 = torch.stack([
            torch.from_numpy(pairs_img1[j]).float().permute(2, 0, 1)
            for j in range(start, end)
        ]).to(device, non_blocking=True)
        batch_img2 = torch.stack([
            torch.from_numpy(pairs_img2[j]).float().permute(2, 0, 1)
            for j in range(start, end)
        ]).to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_fp16):
            flow_out = wrapped_model.calc_flow(batch_img1, batch_img2)

        batch_flows = flow_out['flow'][-1].cpu().numpy()
        for k in range(batch_flows.shape[0]):
            all_flows.append(batch_flows[k])

    # --- Group flows by tgt_long_idx ---
    flows_by_frame = defaultdict(dict)
    for flow_np, (pair_type, tgt_long_idx) in zip(all_flows, pairs_info):
        flows_by_frame[tgt_long_idx][pair_type] = flow_np

    # Optionally attach long flows for visualization
    vis_long_flows = None
    if save_vis:
        vis_long_flows = {}
        for tgt_long_idx, flow_map in flows_by_frame.items():
            if "long" in flow_map:
                vis_long_flows[tgt_long_idx] = flow_map["long"]

    return dict(flows_by_frame), geometry, vis_long_flows


# ===================================================================
# Stage 3: Postprocess -- runs in subprocess
# ===================================================================

def _generate_masks_from_flows(flows_by_frame, geometry, pp_args):
    """
    Pure mask-generation logic shared by postprocess_episode and
    visualization path.  Returns {tgt_long_idx: mask_full_uint8}.
    """
    original_h = geometry['original_h']
    original_w = geometry['original_w']
    x1, y1, x2, y2 = geometry['roi']
    roi_w = geometry['roi_w']
    roi_h = geometry['roi_h']
    scale_factor = geometry['scale_factor']

    compensate = pp_args['compensate_ego_motion']
    ransac_thr = pp_args['ransac_threshold']
    min_thr = pp_args['min_threshold']
    top_pct = pp_args['top_percentile']
    noise_thr = pp_args['noise_threshold']
    short_min_thr = pp_args['short_min_threshold']

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    masks = {}

    for tgt_long_idx, flow_map in flows_by_frame.items():
        try:
            flow_long = flow_map.get("long")
            if flow_long is None:
                continue

            if compensate:
                flow_long = compensate_ego_motion(flow_long, ransac_thr)

            mask_long, _ = generate_robust_motion_mask(
                flow_long, min_threshold=min_thr,
                top_percentile=top_pct, noise_threshold=noise_thr,
            )
            mask_combined = mask_long.copy()

            flow_short_fwd = flow_map.get("short_fwd")
            if flow_short_fwd is not None:
                if compensate:
                    flow_short_fwd = compensate_ego_motion(
                        flow_short_fwd, ransac_thr)
                mask_fwd, _ = generate_robust_motion_mask(
                    flow_short_fwd, min_threshold=short_min_thr,
                    top_percentile=top_pct,
                    noise_threshold=short_min_thr * 5,
                )
                mask_combined = np.clip(mask_combined + mask_fwd, 0, 1)

            flow_short_bwd = flow_map.get("short_bwd")
            if flow_short_bwd is not None:
                if compensate:
                    flow_short_bwd = compensate_ego_motion(
                        flow_short_bwd, ransac_thr)
                mask_bwd, _ = generate_robust_motion_mask(
                    flow_short_bwd, min_threshold=short_min_thr,
                    top_percentile=top_pct,
                    noise_threshold=short_min_thr * 5,
                )
                mask_combined = np.clip(mask_combined + mask_bwd, 0, 1)

            mask_roi = cv2.morphologyEx(
                mask_combined.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

            if scale_factor is not None:
                mask_roi = cv2.resize(mask_roi, (roi_w, roi_h),
                                      interpolation=cv2.INTER_NEAREST)

            mask_full = np.zeros((original_h, original_w), dtype=np.uint8)
            mask_full[y1:y2, x1:x2] = mask_roi
            masks[tgt_long_idx] = mask_full

        except Exception as e:
            print(f"    Postprocess error frame {tgt_long_idx}: {e}")
            continue

    return masks


def postprocess_episode(flows_by_frame, geometry, episode_meta, pp_args,
                        vis_data=None):
    """
    Generate masks from optical flows, RLE-encode, save JSON, and
    optionally save visualizations.
    Runs in a subprocess via ProcessPoolExecutor.

    Args:
        flows_by_frame: {tgt_long_idx: {"long": ndarray, ...}}
        geometry: dict with roi bounds, scale info, original size
        episode_meta: dict with episode_index, camera_key, output_path, etc.
        pp_args: dict with threshold / compensation parameters
        vis_data: None, or dict with frames, vis_long_flows, output_dir,
                  max_vis_frames, time_offset_frames for visualization

    Returns:
        dict with save stats, or None on failure
    """
    masks = _generate_masks_from_flows(flows_by_frame, geometry, pp_args)

    if not masks:
        return None

    episode_info = {
        'episode_index': episode_meta['episode_index'],
        'dataset': episode_meta['dataset_name'],
        'camera_key': episode_meta['camera_key'],
        'robot_type': episode_meta['robot_type'],
        'time_offset': episode_meta['time_offset'],
        'fps': episode_meta['fps'],
        'from_timestamp': episode_meta['from_timestamp'],
        'to_timestamp': episode_meta['to_timestamp'],
        'video_file': episode_meta['video_file'],
        'episode_length': episode_meta['episode_length'],
    }

    stats = save_episode_optical_flow(
        masks, episode_meta['output_path'], episode_info)

    # Visualization (runs in the same subprocess to avoid sending masks back)
    if vis_data is not None and stats is not None:
        try:
            vis_long_flows = vis_data['vis_long_flows']
            frames_dict = vis_data['frames']
            output_dir = Path(vis_data['output_dir'])
            max_vis_frames = vis_data.get('max_vis_frames', 10)
            time_offset_frames = vis_data.get('time_offset_frames', -30)

            x1, y1, x2, y2 = geometry['roi']
            roi_w = geometry['roi_w']
            roi_h = geometry['roi_h']
            scale_factor = geometry['scale_factor']
            orig_h = geometry['original_h']
            orig_w = geometry['original_w']

            flows_dict_for_vis = {}
            for tgt_idx, flow_long in vis_long_flows.items():
                flow_full = np.zeros(
                    (2, orig_h, orig_w), dtype=np.float32)
                if scale_factor is not None:
                    flow_rx = cv2.resize(
                        flow_long[0], (roi_w, roi_h),
                        interpolation=cv2.INTER_LINEAR) / scale_factor
                    flow_ry = cv2.resize(
                        flow_long[1], (roi_w, roi_h),
                        interpolation=cv2.INTER_LINEAR) / scale_factor
                    flow_full[0, y1:y2, x1:x2] = flow_rx
                    flow_full[1, y1:y2, x1:x2] = flow_ry
                else:
                    flow_full[:, y1:y2, x1:x2] = flow_long
                flows_dict_for_vis[tgt_idx] = (flow_full, None)

            save_episode_visualization_sparse(
                frames_dict, masks, flows_dict_for_vis,
                output_dir, episode_meta['camera_key'],
                episode_meta['episode_index'],
                time_offset_frames=time_offset_frames,
                max_vis_frames=max_vis_frames,
            )
        except Exception as e:
            print(f"    Visualization error ep "
                  f"{episode_meta['episode_index']}: {e}")

    return stats


# ===================================================================
# Pipeline orchestrator
# ===================================================================

def run_pipeline(wrapped_model, all_tasks, args):
    """
    3-stage pipeline:
      1. Prefetch pool decodes videos ahead of time
      2. Main thread runs batched GPU inference
      3. Postprocess pool generates masks and saves JSON

    Uses bounded prefetch buffer to limit memory.
    """
    device = args.device
    roi = None
    if args.use_roi:
        roi = (args.roi_x1, args.roi_y1, args.roi_x2, args.roi_y2)

    pp_args = {
        'compensate_ego_motion': args.compensate_ego_motion,
        'ransac_threshold': args.ransac_threshold,
        'min_threshold': args.min_threshold,
        'top_percentile': args.top_percentile,
        'noise_threshold': args.noise_threshold,
        'short_min_threshold': args.short_min_threshold,
    }

    use_fp16 = getattr(args, 'fp16', False)
    prefetch_buffer = args.prefetch_buffer
    num_prefetch = args.num_prefetch_workers
    num_postprocess = args.num_postprocess_workers

    camera_stats = defaultdict(lambda: {
        'total_episodes': 0, 'successful_episodes': 0,
        'failed_episodes': 0, 'total_frames': 0,
        'total_original_size_mb': 0.0, 'total_compressed_size_mb': 0.0,
    })

    total_tasks = len(all_tasks)
    print(f"\nPipeline: {total_tasks} tasks, "
          f"prefetch={num_prefetch} workers (buffer={prefetch_buffer}), "
          f"postprocess={num_postprocess} workers, "
          f"max_flow_batch={args.max_flow_batch}, fp16={use_fp16}")

    spawn_ctx = mp.get_context('spawn')
    prefetch_pool = ProcessPoolExecutor(
        max_workers=num_prefetch, mp_context=spawn_ctx)
    postprocess_pool = ProcessPoolExecutor(
        max_workers=num_postprocess, mp_context=spawn_ctx)

    try:
        # Submit initial batch of prefetch jobs (bounded by prefetch_buffer)
        pending_prefetch = {}  # future -> task_index
        postprocess_futures = []
        task_iter = iter(range(total_tasks))
        submitted_count = 0

        def _submit_prefetch_jobs(n):
            nonlocal submitted_count
            for _ in range(n):
                try:
                    idx = next(task_iter)
                except StopIteration:
                    break
                future = prefetch_pool.submit(prefetch_episode,
                                              all_tasks[idx])
                pending_prefetch[future] = idx
                submitted_count += 1

        _submit_prefetch_jobs(prefetch_buffer)

        pbar = tqdm(total=total_tasks, desc="Processing episodes")
        completed_count = 0

        while pending_prefetch:
            done_futures = []
            for fut in list(pending_prefetch.keys()):
                if fut.done():
                    done_futures.append(fut)

            if not done_futures:
                # Wait for at least one to complete
                done_fut = next(as_completed(pending_prefetch))
                done_futures = [done_fut]

            for fut in done_futures:
                task_idx = pending_prefetch.pop(fut)
                task = all_tasks[task_idx]
                cam_key = task['camera_key']

                # Refill the prefetch buffer
                _submit_prefetch_jobs(1)

                try:
                    episode_data = fut.result()
                except Exception as e:
                    print(f"  Prefetch error task {task_idx}: {e}")
                    camera_stats[cam_key]['total_episodes'] += 1
                    camera_stats[cam_key]['failed_episodes'] += 1
                    completed_count += 1
                    pbar.update(1)
                    continue

                camera_stats[cam_key]['total_episodes'] += 1

                if episode_data is None:
                    # Skipped (already exists or black frame)
                    completed_count += 1
                    pbar.update(1)
                    continue

                # --- Stage 2: GPU inference (main thread) ---
                try:
                    flows_by_frame, geometry, vis_long_flows = \
                        build_pairs_and_infer(
                            wrapped_model, episode_data, device, roi,
                            args.max_flow_batch, args.use_bidirectional,
                            use_fp16, args.save_vis,
                        )
                except Exception as e:
                    print(f"  Inference error ep {episode_data['episode_index']}"
                          f" cam {cam_key}: {e}")
                    traceback.print_exc()
                    camera_stats[cam_key]['failed_episodes'] += 1
                    completed_count += 1
                    pbar.update(1)
                    continue

                if not flows_by_frame:
                    camera_stats[cam_key]['failed_episodes'] += 1
                    completed_count += 1
                    pbar.update(1)
                    continue

                episode_meta = {
                    'episode_index': episode_data['episode_index'],
                    'camera_key': episode_data['camera_key'],
                    'output_path': episode_data['output_path'],
                    'dataset_name': episode_data['dataset_name'],
                    'robot_type': episode_data['robot_type'],
                    'time_offset': args.time_offset,
                    'fps': episode_data['fps'],
                    'from_timestamp': episode_data['from_timestamp'],
                    'to_timestamp': episode_data['to_timestamp'],
                    'video_file': episode_data['video_file'],
                    'episode_length': episode_data['episode_length'],
                }

                # Build vis_data if visualization is requested.
                # Frames and long flows are sent to the postprocess worker
                # so it can generate masks + visualization in one shot.
                vis_data = None
                if args.save_vis and vis_long_flows:
                    offset_in_frames = int(
                        args.time_offset * episode_data['fps'])
                    vis_data = {
                        'frames': episode_data['frames'],
                        'vis_long_flows': vis_long_flows,
                        'output_dir': str(Path(args.output_root)
                                          / episode_data['dataset_name']),
                        'max_vis_frames': args.max_vis_frames,
                        'time_offset_frames': -offset_in_frames,
                    }

                # --- Stage 3: submit postprocessing (non-blocking) ---
                pp_future = postprocess_pool.submit(
                    postprocess_episode,
                    flows_by_frame, geometry, episode_meta, pp_args,
                    vis_data,
                )
                pp_future._v3_cam_key = cam_key

                postprocess_futures.append(pp_future)

                del episode_data

                completed_count += 1
                pbar.update(1)

        pbar.close()

        # --- Collect postprocessing results ---
        print("\nWaiting for postprocessing to finish...")
        for pp_fut in tqdm(postprocess_futures,
                           desc="Collecting postprocess results"):
            cam_key = pp_fut._v3_cam_key
            try:
                stats = pp_fut.result()
                if stats:
                    camera_stats[cam_key]['successful_episodes'] += 1
                    camera_stats[cam_key]['total_frames'] += \
                        stats['num_frames']
                    camera_stats[cam_key]['total_original_size_mb'] += \
                        stats['original_size_mb']
                    camera_stats[cam_key]['total_compressed_size_mb'] += \
                        stats['compressed_size_mb']
                else:
                    camera_stats[cam_key]['failed_episodes'] += 1
            except Exception as e:
                print(f"  Postprocess collect error: {e}")
                traceback.print_exc()
                camera_stats[cam_key]['failed_episodes'] += 1

    finally:
        prefetch_pool.shutdown(wait=False)
        postprocess_pool.shutdown(wait=True)

    return dict(camera_stats)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimized WAFT LeRobot v3.0 optical flow annotation "
                    "(3-stage pipeline: prefetch / batched GPU / postprocess)")

    # WAFT model
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--scale", type=float, default=0.0)

    # Dataset
    parser.add_argument("--data_root", "-d", required=True)
    parser.add_argument("--output_root", "-o", required=True)
    parser.add_argument("--datasets", nargs='+', default=None)

    # Processing (same as v1)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--time_offset", type=float, default=1.0)
    parser.add_argument("--max-fps", type=float, default=None)
    parser.add_argument("--min_threshold", type=float, default=1.0)
    parser.add_argument("--top_percentile", type=float, default=10.0)
    parser.add_argument("--noise_threshold", type=float, default=5.0)
    parser.add_argument("--short_time_offset", type=float, default=0.1)
    parser.add_argument("--use_bidirectional", action="store_true",
                        default=True)
    parser.add_argument("--no_bidirectional", dest="use_bidirectional",
                        action="store_false")
    parser.add_argument("--short_min_threshold", type=float, default=0.5)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--compensate_ego_motion", action="store_true")
    parser.add_argument("--ransac_threshold", type=float, default=3.0)

    # Visualization (same as v1)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--max_vis_frames", type=int, default=10)

    # ROI (same as v1)
    parser.add_argument("--use_roi", action="store_true")
    parser.add_argument("--roi_x1", type=int, default=50)
    parser.add_argument("--roi_y1", type=int, default=0)
    parser.add_argument("--roi_x2", type=int, default=540)
    parser.add_argument("--roi_y2", type=int, default=425)

    # Pipeline performance
    parser.add_argument("--num_prefetch_workers", type=int, default=4,
                        help="Number of processes for video decoding")
    parser.add_argument("--num_postprocess_workers", type=int, default=4,
                        help="Number of processes for mask generation + save")
    parser.add_argument("--prefetch_buffer", type=int, default=8,
                        help="Max episodes pre-decoded in memory")
    parser.add_argument("--max_flow_batch", type=int, default=16,
                        help="Max (img1,img2) pairs per calc_flow call")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Enable AMP FP16 inference")
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")

    args = waft_parse_args(parser)

    # Defaults for attrs that waft_parse_args might not set
    _defaults = {
        'datasets': None, 'time_offset': 1.0, 'max_fps': None,
        'min_threshold': 1.0, 'top_percentile': 10.0,
        'noise_threshold': 5.0, 'skip_existing': False,
        'compensate_ego_motion': False, 'ransac_threshold': 3.0,
        'short_time_offset': 0.1, 'use_bidirectional': True,
        'short_min_threshold': 0.5, 'save_vis': False,
        'max_vis_frames': 10,
        'use_roi': False, 'roi_x1': 50, 'roi_y1': 0,
        'roi_x2': 540, 'roi_y2': 425,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_prefetch_workers': 4, 'num_postprocess_workers': 4,
        'prefetch_buffer': 8, 'max_flow_batch': 16, 'fp16': False,
    }
    for k, v in _defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    print("=" * 80)
    print("WAFT LeRobot v3.0 Optical Flow Annotation (v3 - pipeline)")
    print("=" * 80)
    print(f"WAFT Config: {args.cfg}")
    print(f"WAFT Checkpoint: {args.ckpt}")
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"Device: {args.device}")
    print(f"Prefetch workers: {args.num_prefetch_workers} "
          f"(buffer={args.prefetch_buffer})")
    print(f"Postprocess workers: {args.num_postprocess_workers}")
    print(f"Max flow batch: {args.max_flow_batch}")
    print(f"FP16: {args.fp16}")
    print(f"Time offset: {args.time_offset}s (long) / "
          f"{args.short_time_offset}s (short)")
    print(f"Max fps: "
          f"{args.max_fps if args.max_fps is not None else 'no limit'}")
    print(f"Bidirectional short flow: {args.use_bidirectional}")
    print(f"Scale: {args.scale}")
    print(f"Thresholds: min={args.min_threshold} (long) / "
          f"{args.short_min_threshold} (short), "
          f"top_percentile={args.top_percentile}, "
          f"noise={args.noise_threshold}")
    if args.compensate_ego_motion:
        print(f"Ego-motion compensation: ON "
              f"(RANSAC threshold={args.ransac_threshold}px)")
    if args.use_roi:
        print(f"ROI: ({args.roi_x1}, {args.roi_y1}) to "
              f"({args.roi_x2}, {args.roi_y2})")
    print("=" * 80)

    # Load model
    print("\nLoading WAFT model...")
    model = fetch_model(args)
    load_ckpt(model, args.ckpt)
    model = model.to(args.device)
    model.eval()
    wrapped_model = InferenceWrapper(
        model,
        scale=args.scale,
        train_size=args.image_size,
        pad_to_train_size=False,
        tiling=False,
    )
    print("WAFT model loaded successfully!")

    # Discover datasets
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        dataset_paths = [data_root / name for name in args.datasets]
    else:
        dataset_paths = sorted([
            p for p in data_root.iterdir()
            if p.is_dir() and p.name.startswith('temp_')
        ])
    print(f"\nFound {len(dataset_paths)} datasets to process")

    # Build flat task list: one task per (dataset, episode, camera)
    all_tasks = []
    dataset_infos = {}
    for dp in dataset_paths:
        if not dp.exists():
            print(f"Warning: Dataset not found: {dp}")
            continue
        try:
            ds_info = load_dataset_info(dp)
        except Exception as e:
            print(f"Warning: Failed to load info for {dp}: {e}")
            continue
        dataset_name = dp.name
        dataset_infos[dataset_name] = ds_info
        print(f"  {dataset_name}: {len(ds_info['episodes'])} episodes, "
              f"cameras={ds_info['active_cameras']}")

        for ep in ds_info['episodes']:
            for cam in ds_info['active_cameras']:
                all_tasks.append({
                    'dataset_path': str(dp),
                    'dataset_name': dataset_name,
                    'episode': ep,
                    'camera_key': cam,
                    'fps': ds_info['fps'],
                    'robot_type': ds_info['robot_type'],
                    'time_offset': args.time_offset,
                    'short_time_offset': args.short_time_offset,
                    'max_fps': args.max_fps,
                    'skip_existing': args.skip_existing,
                    'output_root': str(output_root),
                    'use_bidirectional': args.use_bidirectional,
                })

    print(f"\nTotal work items (episode x camera): {len(all_tasks)}")

    if not all_tasks:
        print("Nothing to process.")
        return

    # Run pipeline
    camera_stats = run_pipeline(wrapped_model, all_tasks, args)

    # Print summary
    print("\n" + "=" * 80)
    print("Summary by camera:")
    for cam_key, cs in sorted(camera_stats.items()):
        print(f"\n  Camera {cam_key}:")
        print(f"    Total: {cs['total_episodes']}  "
              f"OK: {cs['successful_episodes']}  "
              f"Failed: {cs['failed_episodes']}  "
              f"Frames: {cs['total_frames']}")
        if cs['total_compressed_size_mb'] > 0:
            ratio = (cs['total_original_size_mb']
                     / cs['total_compressed_size_mb'])
            print(f"    Compression: {cs['total_original_size_mb']:.2f}MB -> "
                  f"{cs['total_compressed_size_mb']:.2f}MB ({ratio:.1f}x)")

    # Save summary JSON
    summary_path = output_root / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'total_tasks': len(all_tasks),
            'cameras': camera_stats,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("Batch processing completed!")
    print(f"Results saved to: {output_root}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
