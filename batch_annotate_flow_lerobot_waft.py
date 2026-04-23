"""
批量标注LeRobot v3.0数据集的光流（使用WAFT模型，1秒间隔）

该脚本使用WAFT模型遍历LeRobot v3.0格式的数据集，对每个episode计算当前帧到1秒后的光流，
生成运动mask并保存为RLE压缩格式。

数据格式：
- 输入：LeRobot v3.0格式数据集（包含meta/info.json和videos/）
- 输出：每个episode每个相机独立的JSON文件（包含RLE压缩的mask）

特性：
1. 使用WAFT光流模型（支持多种配置）
2. 智能相机选择：仅处理cam_high和cam_side（根据robot_type过滤黑帧相机）
3. 时间戳索引：从合并视频中提取特定episode的帧
4. 1秒间隔光流：计算当前帧到1秒后（30帧）的光流
5. RLE压缩：大幅减少存储空间（>10x压缩率）
6. 可选可视化：支持生成调试可视化

Usage:
    python batch_annotate_flow_lerobot_waft.py \
        --data_root /path/to/robochallenge_all_temp/ \
        --output_root /path/to/output \
        --cfg config/a2/twins/tar-c-t.json \
        --ckpt path/to/checkpoint.pth \
        --time_offset 1.0
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from tqdm import tqdm
import traceback
import pandas as pd
import cv2
import numpy as np
import torch

import pycocotools.mask as mask_util

# Add WAFT modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.parser import parse_args as waft_parse_args
from model import fetch_model
from utils.utils import load_ckpt
from inference_tools import InferenceWrapper


from generate_mask import generate_mask_magnitude_threshold, generate_robust_motion_mask
from lerobot.datasets.video_utils import decode_video_frames_torchvision

# 机器人类型与有效相机映射
VALID_CAMERAS = {
    'arx5': ['cam_high', 'cam_side'],
    'ur5': ['cam_high'],  # cam_side是黑帧
    'franka': ['cam_high', 'cam_side'],
    'aloha': ['cam_high'],  # cam_side是黑帧
    'r1lite': ['observation.images.head_rgb'],
    "unitree_g1": ["observation.images.head_stereo_left", "observation.images.head_stereo_right"],
    "widowx": ["observation.images.image_0", "observation.images.image_1", "observation.images.image_2", "observation.images.image_3"],
    "google_robot": ["observation.images.image"]
}

# 目标相机列表（仅处理这两个）
TARGET_CAMERAS = ['cam_high', 'cam_side', 'observation.images.head_rgb', 'observation.images.head_stereo_left', "observation.images.image_0", "observation.images.image_1", "observation.images.image_2", "observation.images.image_3", "observation.images.image"]


def load_dataset_info(dataset_path):
    """
    加载数据集元信息和有效相机列表
    
    Args:
        dataset_path: 数据集根目录
    
    Returns:
        dict: 包含info, robot_type, active_cameras, episodes, fps等信息
    """
    dataset_path = Path(dataset_path)
    
    # 读取info.json
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    robot_type = info.get('robot_type', 'unknown')
    fps = info.get('fps', 30)
    
    # 确定有效相机（仅保留TARGET_CAMERAS中的）
    valid_cams = VALID_CAMERAS.get(robot_type, ['cam_high'])
    active_cameras = [cam for cam in valid_cams if cam in TARGET_CAMERAS]
    
    # 读取所有episodes
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
        'video_path_template': info.get('video_path', 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
    }

def compensate_ego_motion(flow, ransac_reproj_threshold=3.0):
    """
    Remove background ego-motion from optical flow using Homography RANSAC.

    For head-mounted or moving cameras, the background moves globally due to
    camera ego-motion. This fits a homography (global geometric transform) to
    the flow field using RANSAC:
      - Background pixels are inliers (follow the homography)
      - Foreground objects (robot arm, manipulated objects) are outliers

    Returns the residual flow (actual - predicted_background), which contains
    only independent foreground motion.

    Args:
        flow: (2, H, W) optical flow, flow[0]=dx, flow[1]=dy
        ransac_reproj_threshold: RANSAC inlier threshold in pixels (default 3.0)

    Returns:
        residual_flow: (2, H, W) ego-motion compensated flow
    """
    h, w = flow.shape[1:]

    # Subsample on a grid for RANSAC speed (~10000 points is plenty)
    step = max(1, int(math.sqrt(h * w / 10000)))
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    src = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    dst = src + np.stack([
        flow[0, ::step, ::step].ravel(),
        flow[1, ::step, ::step].ravel()
    ], axis=-1).astype(np.float32)

    # Fit homography with RANSAC
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_reproj_threshold)
    if H is None:
        return flow

    # Compute predicted background flow for all pixels using the homography
    # H @ [x, y, 1]^T = [x', y', w']^T  →  pred = (x'/w', y'/w')
    xx_grid, yy_grid = np.meshgrid(
        np.arange(w, dtype=np.float64),
        np.arange(h, dtype=np.float64)
    )
    denom = H[2, 0] * xx_grid + H[2, 1] * yy_grid + H[2, 2]
    pred_x = (H[0, 0] * xx_grid + H[0, 1] * yy_grid + H[0, 2]) / denom
    pred_y = (H[1, 0] * xx_grid + H[1, 1] * yy_grid + H[1, 2]) / denom

    pred_flow = np.zeros_like(flow)
    pred_flow[0] = (pred_x - xx_grid).astype(np.float32)
    pred_flow[1] = (pred_y - yy_grid).astype(np.float32)

    return flow - pred_flow


def compute_needed_frame_indices(total_frames, fps, max_fps, time_offset, short_time_offset):
    """
    Pre-compute the set of original frame indices that will actually be accessed
    during processing. This avoids decoding frames that are never used.

    Returns:
        sorted list of needed frame indices
    """
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
        needed.add(sampled_indices[i])                          # src
        needed.add(sampled_indices[i + offset_frames])          # tgt_long
        short_j = max(0, i + offset_frames - offset_frames_short)
        needed.add(sampled_indices[short_j])                    # tgt_short

    return sorted(needed)


def extract_episode_frames_sparse(dataset_root, episode, camera_key, fps, needed_indices):
    """
    Decode only the specific frames needed from a video file.
    Uses decord for fast random-access batch decoding if available,
    falls back to lerobot's pyav decoder with filtered timestamps.

    Args:
        dataset_root: dataset root path
        episode: episode metadata dict
        camera_key: camera name
        fps: dataset frame rate
        needed_indices: sorted list of frame indices to decode

    Returns:
        dict {frame_idx: np.array (H, W, 3) uint8 RGB}
    """
    chunk_idx = episode[f'videos/{camera_key}/chunk_index']
    file_idx = episode[f'videos/{camera_key}/file_index']
    from_ts = episode[f'videos/{camera_key}/from_timestamp']

    video_path = dataset_root / "videos" / camera_key / \
                 f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    needed_indices = sorted(needed_indices)

    # Decode only needed timestamps via lerobot's pyav decoder
    timestamps = [float(from_ts + idx / fps) for idx in needed_indices]

    frames_tensor = decode_video_frames_torchvision(
        video_path=video_path,
        timestamps=timestamps,
        tolerance_s=1.0 / fps,
        backend="pyav",
        log_loaded_timestamps=False
    )

    # (N, C, H, W) -> (N, H, W, C)
    frames = frames_tensor.permute(0, 2, 3, 1).numpy()
    if frames.dtype == np.float32 and frames.max() <= 1.0:
        frames = (frames * 255).astype(np.uint8)
    elif frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)

    frame_dict = {}
    for i, idx in enumerate(needed_indices):
        if i < len(frames):
            frame_dict[idx] = frames[i]

    return frame_dict


@torch.no_grad()
def process_episode_optical_flow_waft(
    wrapped_model,
    frames,
    total_frames,
    fps=30,
    time_offset=1.0,
    short_time_offset=0.1,
    use_bidirectional=True,
    short_min_threshold=0.5,
    max_fps=None,
    device='cuda',
    min_threshold=1.0,
    top_percentile=10,
    noise_threshold=5.0,
    save_flow_for_vis=False,
    roi=None,
    ego_motion_compensation=False,
    ransac_threshold=3.0
):
    """
    使用WAFT处理单个episode的所有帧对，计算光流并生成mask

    Args:
        wrapped_model: WAFT InferenceWrapper模型
        frames: dict {frame_idx: (H, W, 3) numpy array} 或 (T, H, W, 3) numpy数组
        total_frames: episode总帧数（用于计算采样索引）
        fps: 帧率
        time_offset: 长时间偏移（秒），用于捕捉运动轨迹
        short_time_offset: 短时间偏移（秒），用于低遮挡光流（默认0.1s）
        use_bidirectional: 是否对短偏移使用双向光流（默认True）
        short_min_threshold: 短偏移光流mask的最小阈值（默认0.5，比长偏移低）
        max_fps: 处理帧率上限（Hz）。若fps > max_fps，则降采样，但保持原始frame_id不变。
                 None表示不限制（处理所有帧）。
        device: 计算设备
        min_threshold: 长偏移mask生成的最小阈值
        top_percentile: mask生成的分位数
        noise_threshold: mask生成的噪声阈值
        save_flow_for_vis: 是否保存光流数据用于可视化
        roi: ROI区域 (x1, y1, x2, y2)，只在此区域计算光流

    Returns:
        masks: {frame_idx: mask} 字典（key是原始帧索引）
        flows_dict: {frame_idx: (flow, covisibility)} 字典（如果save_flow_for_vis=True）
    """
    # 降采样：根据max_fps计算stride，保持原始frame_id不变
    if max_fps is not None and max_fps > 0 and fps > max_fps:
        downsample_factor = int(math.ceil(fps / max_fps))
        print(f"    Downsampling from {fps}Hz to ~{fps/downsample_factor:.1f}Hz (factor={downsample_factor}, max_fps={max_fps})")
    else:
        downsample_factor = 1
    sampled_indices = list(range(0, total_frames, downsample_factor))
    effective_fps = fps / downsample_factor

    offset_frames = int(time_offset * effective_fps)
    offset_frames_short = max(1, int(short_time_offset * effective_fps))
    num_valid_frames = len(sampled_indices) - offset_frames

    if num_valid_frames <= 0:
        if save_flow_for_vis:
            return {}, {}
        return {}

    # 获取原始尺寸（从dict的任意一帧获取）
    sample_frame = next(iter(frames.values()))
    original_h, original_w = sample_frame.shape[:2]
    
    # 解析ROI
    if roi is not None:
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(original_w, x2), min(original_h, y2)
        roi_w, roi_h = x2 - x1, y2 - y1
        print(f"    Using ROI: ({x1}, {y1}) to ({x2}, {y2}), size: {roi_w}x{roi_h}")
    else:
        x1, y1, x2, y2 = 0, 0, original_w, original_h
        roi_w, roi_h = original_w, original_h
    
    masks = {}
    flows_dict = {} if save_flow_for_vis else None
    
    # 逐帧处理（WAFT不支持批量，每次处理一对）
    # 使用降采样后的索引，但保存时用原始frame_id
    for i in tqdm(range(num_valid_frames), desc="    Processing frames", leave=False):
        try:
            # 获取原始帧索引
            original_src_idx  = sampled_indices[i]
            original_tgt_long = sampled_indices[i + offset_frames]  # 长偏移目标帧

            # 短偏移目标帧：未来帧的稍早邻帧（t+1s-0.1s），用于低遮挡地捕捉未来位置
            short_j = max(0, i + offset_frames - offset_frames_short)
            original_tgt_short = sampled_indices[short_j]

            src      = frames[original_src_idx]
            tgt_long = frames[original_tgt_long]

            # 裁剪ROI区域
            src_roi      = src[y1:y2, x1:x2]
            tgt_long_roi = tgt_long[y1:y2, x1:x2]

            # 如果最长边 > 640，等比缩放到最长边为640
            h_roi, w_roi = src_roi.shape[:2]
            longest_edge = max(h_roi, w_roi)
            if longest_edge > 640:
                scale_factor = 640 / longest_edge
                new_w = int(w_roi * scale_factor)
                new_h = int(h_roi * scale_factor)
                src_roi      = cv2.resize(src_roi,      (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                tgt_long_roi = cv2.resize(tgt_long_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                scale_factor = None

            # 转换为tensor（RGB -> tensor [1, 3, H, W]）
            image1      = torch.tensor(src_roi,      dtype=torch.float32).permute(2, 0, 1)[None].to(device)
            image2_long = torch.tensor(tgt_long_roi, dtype=torch.float32).permute(2, 0, 1)[None].to(device)

            # WAFT调用1：长偏移反向光流（未来帧→当前帧，mask在未来帧的像素网格上，
            #            标注1s后哪些像素在运动）
            flow_long = wrapped_model.calc_flow(image2_long, image1)['flow'][-1][0].cpu().numpy()

            # 去除背景ego-motion（Homography RANSAC）
            if ego_motion_compensation:
                flow_long = compensate_ego_motion(flow_long, ransac_threshold)

            # 生成长偏移mask
            mask_long, _ = generate_robust_motion_mask(
                flow_long,
                min_threshold=min_threshold,
                top_percentile=top_percentile,
                noise_threshold=noise_threshold
            )
            mask_combined = mask_long.copy()

            # 短偏移处理：围绕未来帧计算，捕捉t+1s处的机械臂位置（低遮挡）
            if original_tgt_short != original_tgt_long:
                tgt_short     = frames[original_tgt_short]
                tgt_short_roi = tgt_short[y1:y2, x1:x2]
                if scale_factor is not None:
                    tgt_short_roi = cv2.resize(tgt_short_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                image2_short  = torch.tensor(tgt_short_roi, dtype=torch.float32).permute(2, 0, 1)[None].to(device)

                # WAFT调用2：短偏移前向光流（未来帧→未来帧+0.1s，定位未来帧中的机械臂）
                flow_short_fwd = wrapped_model.calc_flow(image2_long, image2_short)['flow'][-1][0].cpu().numpy()
                if ego_motion_compensation:
                    flow_short_fwd = compensate_ego_motion(flow_short_fwd, ransac_threshold)
                mask_fwd, _ = generate_robust_motion_mask(
                    flow_short_fwd,
                    min_threshold=short_min_threshold,
                    top_percentile=top_percentile,
                    noise_threshold=short_min_threshold * 5
                )
                mask_combined = np.clip(mask_combined + mask_fwd, 0, 1)

                # WAFT调用3：短偏移反向光流（未来帧+0.1s→未来帧，捕捉机械臂尾缘）
                if use_bidirectional:
                    flow_short_bwd = wrapped_model.calc_flow(image2_short, image2_long)['flow'][-1][0].cpu().numpy()
                    if ego_motion_compensation:
                        flow_short_bwd = compensate_ego_motion(flow_short_bwd, ransac_threshold)
                    mask_bwd, _ = generate_robust_motion_mask(
                        flow_short_bwd,
                        min_threshold=short_min_threshold,
                        top_percentile=top_percentile,
                        noise_threshold=short_min_threshold * 5
                    )
                    mask_combined = np.clip(mask_combined + mask_bwd, 0, 1)

            # 形态学闭运算：连接三个mask之间的小间隙，形成完整的运动区域
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask_roi = cv2.morphologyEx(mask_combined.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

            # 如果缩放过，将mask resize回原始ROI尺寸
            if scale_factor is not None:
                mask_roi = cv2.resize(mask_roi, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

            # 创建完整的mask（原始尺寸），初始化为0（无运动）
            mask_full = np.zeros((original_h, original_w), dtype=np.uint8)
            mask_full[y1:y2, x1:x2] = mask_roi

            # 使用未来帧索引作为key（mask描述的是未来帧的运动区域）
            masks[original_tgt_long] = mask_full

            # 保存光流数据（用于可视化，使用长偏移光流）
            if save_flow_for_vis:
                flow_full = np.zeros((2, original_h, original_w), dtype=np.float32)
                if scale_factor is not None:
                    # flow在缩放分辨率下计算，需要resize回原始ROI尺寸并缩放值
                    flow_resized_x = cv2.resize(flow_long[0], (roi_w, roi_h), interpolation=cv2.INTER_LINEAR) / scale_factor
                    flow_resized_y = cv2.resize(flow_long[1], (roi_w, roi_h), interpolation=cv2.INTER_LINEAR) / scale_factor
                    flow_full[0, y1:y2, x1:x2] = flow_resized_x
                    flow_full[1, y1:y2, x1:x2] = flow_resized_y
                else:
                    flow_full[:, y1:y2, x1:x2] = flow_long
                # WAFT没有covisibility，用None占位
                flows_dict[original_tgt_long] = (flow_full, None)

        except Exception as e:
            print(f"    Error processing frame {original_src_idx}: {e}")
            continue
    
    if save_flow_for_vis:
        return masks, flows_dict
    return masks


def save_episode_optical_flow(masks, output_path, episode_info):
    """
    保存单个episode的光流mask（pycocotools COCO RLE格式）
    """
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
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

    return {
        'original_size_mb': original_size / 1024 / 1024,
        'compressed_size_mb': compressed_size / 1024 / 1024,
        'compression_ratio': compression_ratio,
        'num_frames': len(masks)
    }


def save_episode_visualization_sparse(frames_dict, masks, flows_dict, output_dir, camera_key,
                                       episode_idx, time_offset_frames=30, max_vis_frames=10):
    """
    保存可视化结果（支持sparse frame dict）

    Args:
        frames_dict: {frame_idx: (H,W,3) numpy array} 帧字典
        masks: {frame_idx: mask} mask字典
        flows_dict: {frame_idx: (flow, covisibility)} 光流字典
        output_dir: 输出目录
        camera_key: 相机名称
        episode_idx: episode索引
        time_offset_frames: 时间偏移帧数（负数表示target在source之前）
        max_vis_frames: 最多可视化的帧数
    """
    import flow_vis

    vis_dir = output_dir / "visualizations" / f"episode_{episode_idx:06d}_{camera_key}"
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

        vis_images = []
        vis_images.append(source)
        vis_images.append(target)

        # Mask overlay
        overlay = source.copy().astype(np.float32)
        mask_3ch = np.stack([mask, mask, mask], axis=-1).astype(np.float32)
        overlay = overlay * (1 - mask_3ch * 0.5) + np.array([255, 0, 0], dtype=np.float32) * mask_3ch * 0.5
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        vis_images.append(overlay)

        if flow is not None:
            flow_vis_img = flow_vis.flow_to_color(flow.transpose(1, 2, 0))
            vis_images.append(flow_vis_img)

            flow_magnitude = np.sqrt(flow[0]**2 + flow[1]**2)
            magnitude_normalized = (flow_magnitude / (flow_magnitude.max() + 1e-6) * 255).astype(np.uint8)
            magnitude_colormap = cv2.applyColorMap(magnitude_normalized, cv2.COLORMAP_JET)
            magnitude_colormap = cv2.cvtColor(magnitude_colormap, cv2.COLOR_BGR2RGB)
            vis_images.append(magnitude_colormap)

        # Resize all to same height and concat
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
        cv2.imwrite(str(vis_dir / f"frame_{frame_idx:06d}.jpg"), combined_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])


def process_dataset_waft(dataset_path, output_root, wrapped_model, args):
    """
    使用WAFT处理单个数据集
    
    Args:
        dataset_path: 数据集路径
        output_root: 输出根目录
        wrapped_model: WAFT InferenceWrapper
        args: 命令行参数
    """
    dataset_name = dataset_path.name
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*80}")
    
    try:
        # 加载数据集信息
        dataset_info = load_dataset_info(dataset_path)
        
        print(f"Robot type: {dataset_info['robot_type']}")
        print(f"Active cameras: {dataset_info['active_cameras']}")
        print(f"Total episodes: {len(dataset_info['episodes'])}")
        print(f"FPS: {dataset_info['fps']}")
        
        # 创建输出目录
        dataset_output_dir = output_root / dataset_name
        
        # 统计信息
        dataset_stats = {
            'dataset': dataset_name,
            'robot_type': dataset_info['robot_type'],
            'total_episodes': len(dataset_info['episodes']),
            'cameras': {},
            'errors': []
        }
        
        # 解析ROI
        roi = None
        if args.use_roi:
            roi = (args.roi_x1, args.roi_y1, args.roi_x2, args.roi_y2)
        
        # 处理每个相机
        for camera_key in dataset_info['active_cameras']:
            print(f"\n  Processing camera: {camera_key}")
            
            camera_stats = {
                'total_episodes': 0,
                'successful_episodes': 0,
                'failed_episodes': 0,
                'total_frames': 0,
                'total_original_size_mb': 0,
                'total_compressed_size_mb': 0
            }
            
            # 处理每个episode
            for episode in tqdm(dataset_info['episodes'], desc=f"  {camera_key}", leave=True):
                episode_idx = episode['episode_index']
                episode_length = episode['length']
                
                camera_stats['total_episodes'] += 1
                
                # 检查是否已处理
                output_path = dataset_output_dir / camera_key / f"episode_{episode_idx:06d}.json"
                if output_path.exists() and args.skip_existing:
                    camera_stats['successful_episodes'] += 1
                    continue
                
                try:
                    # 计算需要的帧索引（避免解码所有帧）
                    from_ts = episode[f'videos/{camera_key}/from_timestamp']
                    to_ts = episode[f'videos/{camera_key}/to_timestamp']
                    total_frames = int((to_ts - from_ts) * dataset_info['fps'])

                    needed_indices = compute_needed_frame_indices(
                        total_frames, dataset_info['fps'], args.max_fps,
                        args.time_offset, args.short_time_offset
                    )

                    # 只解码需要的帧
                    frames = extract_episode_frames_sparse(
                        dataset_path, episode, camera_key,
                        fps=dataset_info['fps'],
                        needed_indices=needed_indices
                    )
                    if np.all(frames[needed_indices[0]] < 1):
                        continue

                    if len(frames) == 0:
                        print(f"    Warning: No frames extracted for episode {episode_idx}")
                        camera_stats['failed_episodes'] += 1
                        continue

                    # 使用WAFT计算光流和mask
                    if args.save_vis:
                        masks, flows_dict = process_episode_optical_flow_waft(
                            wrapped_model,
                            frames,
                            total_frames,
                            fps=dataset_info['fps'],
                            time_offset=args.time_offset,
                            short_time_offset=args.short_time_offset,
                            use_bidirectional=args.use_bidirectional,
                            short_min_threshold=args.short_min_threshold,
                            max_fps=args.max_fps,
                            device=args.device,
                            min_threshold=args.min_threshold,
                            top_percentile=args.top_percentile,
                            noise_threshold=args.noise_threshold,
                            save_flow_for_vis=True,
                            roi=roi,
                            ego_motion_compensation=args.compensate_ego_motion,
                            ransac_threshold=args.ransac_threshold
                        )
                    else:
                        masks = process_episode_optical_flow_waft(
                            wrapped_model,
                            frames,
                            total_frames,
                            fps=dataset_info['fps'],
                            time_offset=args.time_offset,
                            short_time_offset=args.short_time_offset,
                            use_bidirectional=args.use_bidirectional,
                            short_min_threshold=args.short_min_threshold,
                            max_fps=args.max_fps,
                            device=args.device,
                            min_threshold=args.min_threshold,
                            top_percentile=args.top_percentile,
                            noise_threshold=args.noise_threshold,
                            save_flow_for_vis=False,
                            roi=roi,
                            ego_motion_compensation=args.compensate_ego_motion,
                            ransac_threshold=args.ransac_threshold
                        )
                        flows_dict = None

                    if len(masks) == 0:
                        print(f"    Warning: No masks generated for episode {episode_idx}")
                        camera_stats['failed_episodes'] += 1
                        continue

                    # 构造episode_info
                    sample_frame = next(iter(frames.values()))
                    original_h, original_w = sample_frame.shape[:2]
                    episode_info = {
                        'episode_index': episode_idx,
                        'dataset': dataset_name,
                        'camera_key': camera_key,
                        'robot_type': dataset_info['robot_type'],
                        'time_offset': args.time_offset,
                        'fps': dataset_info['fps'],
                        'from_timestamp': episode[f'videos/{camera_key}/from_timestamp'],
                        'to_timestamp': episode[f'videos/{camera_key}/to_timestamp'],
                        'video_file': f"chunk-{episode[f'videos/{camera_key}/chunk_index']:03d}/file-{episode[f'videos/{camera_key}/file_index']:03d}.mp4",
                        'episode_length': episode_length,
                    }
                    
                    # 保存RLE JSON
                    stats = save_episode_optical_flow(masks, output_path, episode_info)
                    
                    if stats:
                        camera_stats['successful_episodes'] += 1
                        camera_stats['total_frames'] += stats['num_frames']
                        camera_stats['total_original_size_mb'] += stats['original_size_mb']
                        camera_stats['total_compressed_size_mb'] += stats['compressed_size_mb']
                    
                    # 保存可视化
                    # mask/flow的key已经是未来帧索引，无需remap
                    if args.save_vis and flows_dict is not None:
                        offset_in_frames = int(args.time_offset * dataset_info['fps'])
                        save_episode_visualization_sparse(
                            frames,
                            masks,
                            flows_dict,
                            dataset_output_dir,
                            camera_key,
                            episode_idx,
                            time_offset_frames=-offset_in_frames,
                            max_vis_frames=args.max_vis_frames
                        )
                    
                except Exception as e:
                    print(f"    Error processing episode {episode_idx}: {e}")
                    traceback.print_exc()
                    camera_stats['failed_episodes'] += 1
                    dataset_stats['errors'].append({
                        'episode': episode_idx,
                        'camera': camera_key,
                        'error': str(e)
                    })
                    continue
            
            # 保存相机统计
            dataset_stats['cameras'][camera_key] = camera_stats
            
            # 打印相机统计
            print(f"\n  Camera {camera_key} Summary:")
            print(f"    Total episodes: {camera_stats['total_episodes']}")
            print(f"    Successful: {camera_stats['successful_episodes']}")
            print(f"    Failed: {camera_stats['failed_episodes']}")
            print(f"    Total frames: {camera_stats['total_frames']}")
            if camera_stats['total_compressed_size_mb'] > 0:
                compression_ratio = camera_stats['total_original_size_mb'] / camera_stats['total_compressed_size_mb']
                print(f"    Compression: {camera_stats['total_original_size_mb']:.2f}MB -> {camera_stats['total_compressed_size_mb']:.2f}MB ({compression_ratio:.1f}x)")
        
        return dataset_stats
        
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        traceback.print_exc()
        return {
            'dataset': dataset_name,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="批量标注LeRobot v3.0数据集的光流（使用WAFT模型）")
    
    # WAFT模型参数
    parser.add_argument("--cfg", required=True, help="WAFT配置文件路径")
    parser.add_argument("--ckpt", required=True, help="WAFT checkpoint路径")
    parser.add_argument("--scale", type=float, default=0.0, help="scale factor for input images")
    
    # 数据集参数
    parser.add_argument("--data_root", "-d", required=True, help="数据根目录路径（包含多个temp_*数据集）")
    parser.add_argument("--output_root", "-o", required=True, help="输出根目录路径")
    parser.add_argument("--datasets", nargs='+', default=None, help="指定要处理的数据集名称（默认：处理所有temp_*数据集）")
    
    # 处理参数
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备（默认：cuda）")
    parser.add_argument("--time_offset", type=float, default=1.0, help="时间偏移（秒），默认1.0秒")
    parser.add_argument("--max-fps", type=float, default=None,
                        help="处理帧率上限（Hz）。若视频fps > max-fps，则降采样，"
                             "但输出的frame_index仍为原始视频索引。（默认：不限制，处理所有帧）")
    parser.add_argument("--min_threshold", type=float, default=1.0, help="Mask生成的最小阈值（默认：1.0）")
    parser.add_argument("--top_percentile", type=float, default=10.0, help="Mask生成的分位数（默认：10）")
    parser.add_argument("--noise_threshold", type=float, default=5.0, help="Mask生成的噪声阈值（默认：5.0）")
    parser.add_argument("--short_time_offset", type=float, default=0.1, help="短时间偏移（秒），用于低遮挡光流（默认：0.1）")
    parser.add_argument("--use_bidirectional", action="store_true", default=True, help="对短偏移使用双向光流（默认：True）")
    parser.add_argument("--no_bidirectional", dest="use_bidirectional", action="store_false", help="禁用双向短时间光流")
    parser.add_argument("--short_min_threshold", type=float, default=0.5, help="短偏移光流mask的最小阈值（默认：0.5）")
    parser.add_argument("--skip_existing", action="store_true", help="跳过已存在的输出文件")

    # Ego-motion compensation
    parser.add_argument("--compensate_ego_motion", action="store_true",
                        help="Enable homography RANSAC ego-motion compensation. "
                             "Removes background motion from moving/head-mounted cameras, "
                             "keeping only foreground (robot + manipulated object) motion.")
    parser.add_argument("--ransac_threshold", type=float, default=3.0,
                        help="RANSAC reprojection threshold in pixels for ego-motion compensation (default: 3.0)")
    
    # 可视化参数
    parser.add_argument("--save_vis", action="store_true", help="保存可视化结果")
    parser.add_argument("--max_vis_frames", type=int, default=10, help="每个episode最多可视化的帧数（默认：10）")
    
    # ROI参数
    parser.add_argument("--use_roi", action="store_true", help="使用ROI区域计算光流")
    parser.add_argument("--roi_x1", type=int, default=50, help="ROI左上角X坐标（默认：50）")
    parser.add_argument("--roi_y1", type=int, default=0, help="ROI左上角Y坐标（默认：0）")
    parser.add_argument("--roi_x2", type=int, default=540, help="ROI右下角X坐标（默认：540）")
    parser.add_argument("--roi_y2", type=int, default=425, help="ROI右下角Y坐标（默认：425）")
    
    # 解析参数（使用WAFT的parse_args来加载cfg）
    args = waft_parse_args(parser)
    
    # 确保自定义参数存在（waft_parse_args可能没有这些字段）
    if not hasattr(args, 'datasets'):
        args.datasets = None
    if not hasattr(args, 'data_root'):
        raise ValueError("Missing required argument: --data_root")
    if not hasattr(args, 'output_root'):
        raise ValueError("Missing required argument: --output_root")
    if not hasattr(args, 'time_offset'):
        args.time_offset = 1.0
    if not hasattr(args, 'max_fps'):
        args.max_fps = None
    if not hasattr(args, 'min_threshold'):
        args.min_threshold = 1.0
    if not hasattr(args, 'top_percentile'):
        args.top_percentile = 10.0
    if not hasattr(args, 'noise_threshold'):
        args.noise_threshold = 5.0
    if not hasattr(args, 'skip_existing'):
        args.skip_existing = False
    if not hasattr(args, 'compensate_ego_motion'):
        args.compensate_ego_motion = False
    if not hasattr(args, 'ransac_threshold'):
        args.ransac_threshold = 3.0
    if not hasattr(args, 'short_time_offset'):
        args.short_time_offset = 0.1
    if not hasattr(args, 'use_bidirectional'):
        args.use_bidirectional = True
    if not hasattr(args, 'short_min_threshold'):
        args.short_min_threshold = 0.5
    if not hasattr(args, 'save_vis'):
        args.save_vis = False
    if not hasattr(args, 'max_vis_frames'):
        args.max_vis_frames = 10
    if not hasattr(args, 'use_roi'):
        args.use_roi = False
    if not hasattr(args, 'roi_x1'):
        args.roi_x1 = 50
    if not hasattr(args, 'roi_y1'):
        args.roi_y1 = 0
    if not hasattr(args, 'roi_x2'):
        args.roi_x2 = 540
    if not hasattr(args, 'roi_y2'):
        args.roi_y2 = 425
    if not hasattr(args, 'device'):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*80)
    print("WAFT LeRobot v3.0 Optical Flow Annotation")
    print("="*80)
    print(f"WAFT Config: {args.cfg}")
    print(f"WAFT Checkpoint: {args.ckpt}")
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"Device: {args.device}")
    print(f"Time offset: {args.time_offset}s (long) / {args.short_time_offset}s (short)")
    print(f"Max fps: {args.max_fps if args.max_fps is not None else 'no limit (all frames)'}")
    print(f"Bidirectional short flow: {args.use_bidirectional}")
    print(f"Scale: {args.scale}")
    print(f"Thresholds: min={args.min_threshold} (long) / {args.short_min_threshold} (short), top_percentile={args.top_percentile}, noise={args.noise_threshold}")
    if args.compensate_ego_motion:
        print(f"Ego-motion compensation: ON (RANSAC threshold={args.ransac_threshold}px)")
    if args.use_roi:
        print(f"ROI: ({args.roi_x1}, {args.roi_y1}) to ({args.roi_x2}, {args.roi_y2})")
    print("="*80)
    
    # 加载WAFT模型
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
        tiling=False
    )
    print("WAFT model loaded successfully!")
    
    # 准备数据集列表
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    if args.datasets:
        dataset_paths = [data_root / name for name in args.datasets]
    else:
        # 默认处理所有temp_*数据集
        dataset_paths = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('temp_')])
    
    print(f"\nFound {len(dataset_paths)} datasets to process")
    
    # 处理每个数据集
    all_stats = []
    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            print(f"Warning: Dataset not found: {dataset_path}")
            continue
        
        stats = process_dataset_waft(dataset_path, output_root, wrapped_model, args)
        all_stats.append(stats)
    
    # 保存总体统计
    summary_path = output_root / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'total_datasets': len(all_stats),
            'datasets': all_stats
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("Batch processing completed!")
    print(f"Results saved to: {output_root}")
    print(f"Summary saved to: {summary_path}")
    print("="*80)


if __name__ == '__main__':
    main()
