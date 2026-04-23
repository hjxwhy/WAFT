"""
从光流文件生成运动区域的binary mask。

该脚本读取batch_inference_flow.py生成的光流文件（.npz格式），
计算光流模长，生成运动区域的binary mask，并抑制背景噪声。

方法说明：
1. 找出大于min_threshold像素的光流模长
2. 计算这些区域的top_percentile%分位数作为阈值
3. 保留模长 > top_percentile%分位数 或者 模长 > noise_threshold像素的区域
4. 这样可以保留明显的运动，同时抑制小的背景噪声

存储格式：
- 默认使用RLE（Run-Length Encoding）压缩，每个序列的所有mask存储在一个JSON文件中
- 文件格式：masks_rle.json，包含所有帧的RLE编码mask
- 可以使用load_compressed_masks()函数加载压缩的mask文件

Usage:
    python generate_mask.py --input_root /path/to/flow/output --output_root /path/to/mask/output
    python generate_mask.py --input_root /path/to/flow/output --min_threshold 1.0 --top_percentile 10 --noise_threshold 5.0
    python generate_mask.py --input_root /path/to/flow/output --output_root /path/to/mask/output --no_rle  # 不使用压缩
"""

import argparse
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

import cv2
import flow_vis
import numpy as np
import shutil



def load_flow_data(npz_path):
    """
    加载光流数据。
    
    Args:
        npz_path: .npz文件路径
    
    Returns:
        flow: (2, H, W) numpy数组，flow[0]是x方向，flow[1]是y方向
        covisibility: (H, W) numpy数组，值在0-1之间
    """
    data = np.load(npz_path)
    flow = data['flow']  # (2, H, W)
    covisibility = data['covisibility']  # (H, W)
    return flow, covisibility


def compute_flow_magnitude(flow):
    """
    计算光流模长。
    
    Args:
        flow: (2, H, W) numpy数组
    
    Returns:
        magnitude: (H, W) numpy数组，光流模长
    """
    flow_x = flow[0]  # (H, W)
    flow_y = flow[1]  # (H, W)
    magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)
    return magnitude


def generate_mask_magnitude_threshold(flow, min_threshold=1.0, top_percentile=10, noise_threshold=5.0):
    """
    基于光流模长阈值生成mask，抑制背景噪声。
    
    步骤：
    1. 找出大于min_threshold像素的光流模长
    2. 计算这些区域的top_percentile%分位数作为阈值
    3. 保留模长 > top_percentile%分位数 或者 模长 > noise_threshold像素的区域
    4. 这样可以保留明显的运动，同时抑制小的背景噪声
    
    Args:
        flow: (2, H, W) numpy数组
        min_threshold: 最小模长阈值（像素），默认1.0
        top_percentile: 分位数（0-100），默认10（即10%分位数）
        noise_threshold: 噪声阈值（像素），小于此值且小于分位数阈值的区域会被去掉，默认5.0
    
    Returns:
        mask: (H, W) binary mask，1表示运动，0表示静止
        info: dict，包含使用的阈值信息
    """
    magnitude = compute_flow_magnitude(flow)
    
    # 步骤1: 找出大于min_threshold的区域
    candidate_mask = magnitude > min_threshold
    
    if not np.any(candidate_mask):
        # 如果没有候选区域，返回全0的mask
        return np.zeros_like(magnitude, dtype=np.uint8), {
            'min_threshold': min_threshold,
            'percentile_threshold': 0,
            'noise_threshold': noise_threshold,
            'final_mask_count': 0
        }
    
    # 步骤2: 计算候选区域的top_percentile%分位数作为阈值
    candidate_magnitudes = magnitude[candidate_mask]
    percentile_threshold = np.percentile(candidate_magnitudes, top_percentile)
    
    # 步骤3: 生成mask
    # 保留模长 > percentile_threshold 或者 模长 > noise_threshold 的区域
    mask = ((magnitude > percentile_threshold) | (magnitude > noise_threshold)).astype(np.uint8)
    
    # 但是只保留那些原本大于min_threshold的区域
    mask = mask & candidate_mask.astype(np.uint8)
    
    info = {
        'min_threshold': min_threshold,
        'percentile_threshold': float(percentile_threshold),
        'noise_threshold': noise_threshold,
        'final_mask_count': int(np.sum(mask))
    }
    
    return mask, info

def generate_robust_motion_mask(flow, min_area=100, use_otsu=True, min_threshold=1.0, top_percentile=10, noise_threshold=5.0):
    """
    针对静止和运动场景都鲁棒的Mask生成。
    
    Args:
        flow: (2, H, W) 光流
        min_area: 最小连通域面积
        motion_floor: 物理底噪（像素）。任何小于此值的运动，即使被Otsu选中，也强制视为静止。
                      通常设为 0.5 到 2.0 之间，取决于你的相机抖动程度。
    """
    magnitude = compute_flow_magnitude(flow)
    
    # --- 防线 1: 全局静止检测 (快速失败机制) ---
    # 如果画面中最大的运动都小于底噪，直接返回全黑，节省计算时间
    # 使用 99% 分位数比 max 更抗单点噪点
    if np.max(magnitude) < min_threshold:
        return np.zeros(magnitude.shape, dtype=np.uint8), None

    # 归一化用于 Otsu
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 计算 Otsu 阈值 (统计阈值)
    otsu_thresh_val, _ = cv2.threshold(mag_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # --- 防线 2: 双重门限 (逻辑与) ---
    # 区域必须满足：
    # 1. (mag_norm > otsu_thresh_val): 在统计分布上属于“强信号”一侧
    # 2. (magnitude > min_threshold): 在物理意义上确实动了（克服传感器底噪）
    
    # 注意：我们需要把 mag_norm 的阈值判断 和 原始 magnitude 的阈值判断结合
    # 因为 otsu_thresh_val 是基于 0-255 的，不好直接对应回像素值，
    # 所以最好的办法是分别生成两个 mask 取交集。
    
    mask_otsu = (mag_norm > otsu_thresh_val)
    mask_floor = (magnitude > min_threshold)
    
    # 核心逻辑：只有通过了统计门槛 AND 物理门槛的才是真运动
    combined_mask = (mask_otsu & mask_floor).astype(np.uint8) * 255
    
    # --- 后处理 (形态学 + 连通域) ---
    # 即使是静止场景，偶尔也有几个噪点能突破双重门限，靠形态学消灭它们
    if np.sum(combined_mask) == 0:
        return np.zeros(magnitude.shape, dtype=np.uint8), None

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    mask_clean = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)
    
    # 连通域过滤
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    final_mask = np.zeros_like(mask_clean)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            final_mask[labels == i] = 1 # 输出 0/1 mask
    info = {
        'min_threshold': min_threshold,
        'noise_threshold': noise_threshold,
        'percentile_threshold': float(0),
        'final_mask_count': int(np.sum(final_mask))
    }
    return final_mask, info

def load_image(image_path):
    """加载图像。"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode_rle(mask):
    """
    将二值mask编码为RLE格式（COCO风格）。
    
    Args:
        mask: (H, W) 二值mask，值为0或1
    
    Returns:
        rle: dict，包含'counts'和'size'
    """
    # 将mask展平为一维数组
    pixels = mask.flatten()
    total_pixels = len(pixels)
    
    # 如果mask全为0
    if np.all(pixels == 0):
        return {
            'counts': [total_pixels],
            'size': [mask.shape[0], mask.shape[1]]
        }
    
    # 如果mask全为1
    if np.all(pixels == 1):
        return {
            'counts': [0, total_pixels],
            'size': [mask.shape[0], mask.shape[1]]
        }
    
    # 添加边界0，确保以0开始和结束
    pixels_with_boundary = np.concatenate([[0], pixels, [0]])
    
    # 找到值变化的位置
    # 例如：[0, 0, 0, 1, 1, 1, 0, 0] -> 添加边界后 [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    # 变化位置：[4, 7] -> 长度应该是 [4, 3, 3] (4个0, 3个1, 3个0)
    change_indices = np.where(pixels_with_boundary[1:] != pixels_with_boundary[:-1])[0] + 1
    
    # 计算每个run的长度
    # 添加开始和结束位置
    run_positions = np.concatenate([[0], change_indices, [len(pixels_with_boundary)]])
    counts = np.diff(run_positions).tolist()
    
    return {
        'counts': counts,
        'size': [mask.shape[0], mask.shape[1]]
    }


def decode_rle(rle):
    """
    将RLE格式解码为二值mask。
    
    Args:
        rle: dict，包含'counts'和'size'
    
    Returns:
        mask: (H, W) 二值mask，值为0或1
    """
    h, w = rle['size']
    counts = rle['counts']
    total_pixels = h * w
    
    # 特殊情况：全0
    if len(counts) == 1:
        if counts[0] == total_pixels:
            return np.zeros((h, w), dtype=np.uint8)
    
    # 特殊情况：全1
    if len(counts) == 2 and counts[0] == 0:
        if counts[1] == total_pixels:
            return np.ones((h, w), dtype=np.uint8)
    
    # 重建像素数组（包含边界）
    pixels_with_boundary = []
    value = 0  # 从0开始（因为编码时添加了边界0）
    for count in counts:
        pixels_with_boundary.extend([value] * count)
        value = 1 - value  # 在0和1之间切换
    
    # 移除边界（第一个和最后一个0）
    # 编码时添加了[0, ...pixels..., 0]，所以解码时要移除第一个和最后一个
    if len(pixels_with_boundary) >= 2:
        pixels = pixels_with_boundary[1:-1]
    else:
        pixels = []
    
    # 确保长度正确
    if len(pixels) != total_pixels:
        # 如果长度不匹配，可能是边界处理有问题，尝试直接重建
        pixels = []
        value = 0
        for i, count in enumerate(counts):
            if i == 0 and count > 0:
                # 第一个count可能是边界0的长度，跳过
                continue
            pixels.extend([value] * count)
            value = 1 - value
        
        # 如果还是不对，填充或截断
        if len(pixels) < total_pixels:
            pixels.extend([0] * (total_pixels - len(pixels)))
        elif len(pixels) > total_pixels:
            pixels = pixels[:total_pixels]
    
    # 重塑为原始形状
    mask = np.array(pixels, dtype=np.uint8).reshape(h, w)
    
    return mask


def load_compressed_masks(rle_file_path):
    """
    加载RLE压缩的mask文件。
    
    Args:
        rle_file_path: RLE JSON文件路径
    
    Returns:
        masks_dict: dict，{frame_idx: mask}，mask为(H, W) numpy数组
        metadata: dict，包含元数据信息
    """
    rle_file_path = Path(rle_file_path)
    
    with open(rle_file_path, 'r') as f:
        rle_data = json.load(f)
    
    masks_dict = {}
    for frame_idx_str, rle in rle_data['frames'].items():
        frame_idx = int(frame_idx_str)
        mask = decode_rle(rle)
        masks_dict[frame_idx] = mask
    
    return masks_dict, rle_data.get('metadata', {})


def save_visualization(source_image, target_image, flow, magnitude, mask, covisibility, 
                      output_path, frame_idx, method_name, threshold_info=""):
    """
    保存可视化结果。
    
    布局：
    上排：source图像 | target图像 | 光流可视化
    中排：光流模长热力图 | covisibility | mask
    下排：source+mask叠加 | target+mask叠加 | mask单独显示
    """
    vis_dir = Path(output_path).parent / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保所有图像尺寸一致
    h, w = source_image.shape[:2]
    target_image = cv2.resize(target_image, (w, h)) if target_image.shape[:2] != (h, w) else target_image
    
    # 1. 光流可视化
    flow_vis_img = flow_vis.flow_to_color(flow.transpose(1, 2, 0))
    
    # 2. 光流模长热力图
    magnitude_normalized = (magnitude / (magnitude.max() + 1e-6) * 255).astype(np.uint8)
    magnitude_colormap = cv2.applyColorMap(magnitude_normalized, cv2.COLORMAP_JET)
    magnitude_colormap = cv2.cvtColor(magnitude_colormap, cv2.COLOR_BGR2RGB)
    
    # 3. covisibility可视化
    covis_vis = (covisibility * 255).astype(np.uint8)
    covis_vis_rgb = np.stack([covis_vis, covis_vis, covis_vis], axis=2)
    
    # 4. mask可视化（RGB）
    mask_rgb = np.stack([mask * 255, mask * 255, mask * 255], axis=2)
    
    # 5. 叠加显示：图像 + mask overlay
    mask_overlay = mask[..., None].astype(np.float32)
    overlay_color = np.array([0, 255, 0], dtype=np.float32)  # 绿色
    source_with_mask = (source_image * (1 - mask_overlay * 0.5) + 
                       overlay_color * mask_overlay * 0.5).astype(np.uint8)
    target_with_mask = (target_image * (1 - mask_overlay * 0.5) + 
                        overlay_color * mask_overlay * 0.5).astype(np.uint8)
    
    # 调整所有图像到相同高度
    def resize_to_height(img, target_h):
        h, w = img.shape[:2]
        if h == target_h:
            return img
        new_w = int(w * target_h / h)
        return cv2.resize(img, (new_w, target_h))
    
    target_h = h
    source_resized = resize_to_height(source_image, target_h)
    target_resized = resize_to_height(target_image, target_h)
    flow_vis_resized = resize_to_height(flow_vis_img, target_h)
    magnitude_resized = resize_to_height(magnitude_colormap, target_h)
    covis_resized = resize_to_height(covis_vis_rgb, target_h)
    mask_resized = resize_to_height(mask_rgb, target_h)
    source_mask_resized = resize_to_height(source_with_mask, target_h)
    target_mask_resized = resize_to_height(target_with_mask, target_h)
    
    # 上排：source | target | flow
    top_row = np.hstack([source_resized, target_resized, flow_vis_resized])
    
    # 中排：magnitude | covisibility | mask
    mid_row = np.hstack([magnitude_resized, covis_resized, mask_resized])
    
    # 下排：source+mask | target+mask | mask only
    bottom_row = np.hstack([source_mask_resized, target_mask_resized, mask_resized])
    
    # 垂直拼接
    combined_vis = np.vstack([top_row, mid_row, bottom_row])
    
    # 添加文本信息
    if threshold_info:
        cv2.putText(combined_vis, f"Method: {method_name} | {threshold_info}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 保存
    vis_path = vis_dir / f"mask_vis_{frame_idx:04d}_{method_name}.png"
    cv2.imwrite(str(vis_path), cv2.cvtColor(combined_vis, cv2.COLOR_RGB2BGR))
    
    return vis_path


def process_flow_file(npz_path, source_image_path, target_image_path, output_dir, 
                     method='magnitude_threshold', **method_kwargs):
    """
    处理单个光流文件，生成mask并保存可视化。
    
    Args:
        npz_path: .npz文件路径
        source_image_path: 源图像路径（可以是None）
        target_image_path: 目标图像路径（可以是None）
        output_dir: 输出目录
        method: mask生成方法（已废弃，保留以兼容）
        **method_kwargs: 方法参数
            - min_threshold: 最小模长阈值（默认1.0）
            - top_percentile: 前百分之几的模长（默认10）
            - noise_threshold: 噪声阈值（默认5.0）
    """
    # 加载数据
    flow, covisibility = load_flow_data(npz_path)
    
    # 检查图像路径
    if source_image_path is None or target_image_path is None:
        source_image = None
        target_image = None
    else:
        source_image = load_image(source_image_path)
        target_image = load_image(target_image_path)
        if source_image is None or target_image is None:
            print(f"Warning: Could not load images for {npz_path}")
            print(f"  source_image_path: {source_image_path}")
            print(f"  target_image_path: {target_image_path}")
            source_image = None
            target_image = None
    
    # 计算光流模长
    magnitude = compute_flow_magnitude(flow)
    
    # 生成mask（只使用magnitude_threshold方法）
    min_threshold = method_kwargs.get('min_threshold', 1.0)
    top_percentile = method_kwargs.get('top_percentile', 10)
    noise_threshold = method_kwargs.get('noise_threshold', 5.0)
    
    # mask, mask_info = generate_mask_magnitude_threshold(
    #     flow, 
    #     min_threshold=min_threshold,
    #     top_percentile=top_percentile,
    #     noise_threshold=noise_threshold
    # )
    mask, mask_info = generate_robust_motion_mask(
        flow, 
        min_threshold=min_threshold,
        top_percentile=top_percentile,
        noise_threshold=noise_threshold
    )
    
    threshold_info = (f"min_thresh={min_threshold}, "
                     f"{top_percentile}%_percentile_thresh={mask_info['percentile_threshold']:.2f}, "
                     f"noise_thresh={noise_threshold}, "
                     f"mask_pixels={mask_info['final_mask_count']}")
    
    frame_idx = int(Path(npz_path).stem.split('_')[-1])
    
    # 保存可视化（只有在图像存在时才保存）
    if method_kwargs.get('save_vis', True) and source_image is not None and target_image is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        mask_path = output_path / f"mask_{frame_idx:04d}.png"
        save_visualization(
            source_image, target_image, flow, magnitude, mask, covisibility,
            mask_path, frame_idx, 'magnitude_threshold', threshold_info
        )
    
    return frame_idx, mask


def find_image_files_from_flow_path(npz_path, frame_idx, frame_interval=1):
    """
    根据光流文件路径查找对应的图像文件。
    
    逻辑：
    - 光流路径：train/flows/bridge/image_3/19914/data/data_0001.npz
    - 图像路径：train/images/bridge/image_3/19914/0001.png
    
    Args:
        npz_path: .npz文件路径
        frame_idx: source帧的索引
        frame_interval: 帧间隔（target = source + frame_interval）
    
    Returns:
        source_file, target_file: 图像文件路径，如果找不到则返回None
    """
    npz_path = Path(npz_path)
    
    # 找到train目录
    current_path = npz_path
    train_dir = None
    flows_dir = None
    
    while current_path != current_path.root:
        if current_path.name == "train" and current_path.is_dir():
            train_dir = current_path
            flows_dir = current_path / "flows"
            if flows_dir.exists():
                break
        current_path = current_path.parent
    
    if train_dir is None or flows_dir is None:
        return None, None
    
    # 计算从flows_dir到video_seq_dir的相对路径
    # npz_path: train/flows/bridge/image_3/19914/data/data_0001.npz
    # video_seq_dir: train/flows/bridge/image_3/19914
    # relative_path: bridge/image_3/19914
    try:
        video_seq_dir = npz_path.parent.parent  # 去掉data和文件名
        relative_path = video_seq_dir.relative_to(flows_dir)
    except ValueError:
        return None, None
    
    # 构建图像目录路径
    images_dir = train_dir / "images"
    image_seq_dir = images_dir / relative_path
    
    if not image_seq_dir.exists():
        return None, None
    
    # 查找source和target图像文件
    source_file = image_seq_dir / f"{frame_idx:04d}.png"
    target_frame_idx = frame_idx + frame_interval
    target_file = image_seq_dir / f"{target_frame_idx:04d}.png"
    
    # 如果png不存在，尝试jpg
    if not source_file.exists():
        source_file = image_seq_dir / f"{frame_idx:04d}.jpg"
    if not target_file.exists():
        target_file = image_seq_dir / f"{target_frame_idx:04d}.jpg"
    
    # 检查文件是否存在
    if source_file.exists() and target_file.exists():
        return source_file, target_file
    
    return None, None


def process_data_directory(input_root, output_root, method='magnitude_threshold', 
                          skip_existing=False, **method_kwargs):
    """
    批量处理数据目录下的所有光流文件。
    
    Args:
        input_root: 输入根目录（包含flow输出的目录）
        output_root: 输出根目录
        method: mask生成方法
        skip_existing: 如果输出文件已存在，是否跳过
        **method_kwargs: 方法参数
    """
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    # 查找所有data目录
    data_dirs = list(input_path.rglob("data"))
    
    if not data_dirs:
        print(f"No 'data' directories found in {input_root}")
        return
    
    print(f"Found {len(data_dirs)} data directories")
    
    for data_dir in data_dirs:
        # 获取对应的视频序列目录（data目录的父目录）
        video_seq_dir = data_dir.parent
        
        # 查找所有.npz文件
        npz_files = sorted(data_dir.glob("*.npz"))
        
        if not npz_files:
            continue
        
        print(f"\nProcessing {len(npz_files)} flow files in {video_seq_dir}")
        
        # 创建对应的输出目录
        relative_path = video_seq_dir.relative_to(input_path)
        output_seq_dir = output_path / relative_path / "masks"
        if output_seq_dir.exists():
            shutil.rmtree(output_seq_dir)
        output_seq_dir = output_path / relative_path
        output_seq_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查压缩文件是否存在
        compressed_file = output_seq_dir / "masks_rle.json"
        if skip_existing and compressed_file.exists():
            print(f"  Skipping {video_seq_dir.name} (compressed file exists)")
            continue
        
        # 收集所有mask
        masks_dict = {}
        frame_indices = []
        
        for npz_file in tqdm(npz_files, desc=f"Processing {video_seq_dir.name}"):
            # 提取frame_idx
            frame_idx = int(npz_file.stem.split('_')[-1])
            frame_indices.append(frame_idx)
            
            # 只在需要可视化时才查找图像文件
            source_file, target_file = None, None
            if method_kwargs.get('save_vis', True):
                # 尝试不同的frame_interval（1, 2, 3, 5, 6）
                for frame_interval in [1, 2, 3, 5, 6]:
                    source_file, target_file = find_image_files_from_flow_path(
                        npz_file, frame_idx, frame_interval
                    )
                    if source_file is not None and target_file is not None:
                        break
            
            # 更新method_kwargs，如果找不到图像则不保存可视化
            method_kwargs_local = method_kwargs.copy()
            method_kwargs_local['save_vis'] = (
                method_kwargs.get('save_vis', True) and 
                source_file is not None and 
                target_file is not None
            )
            
            try:
                frame_idx_result, mask = process_flow_file(
                    npz_file, source_file, target_file, output_seq_dir,
                    method=method, **method_kwargs_local
                )
                masks_dict[frame_idx_result] = mask
            except Exception as e:
                print(f"Error processing {npz_file}: {e}")
                continue
        
        # 保存压缩的mask文件
        if masks_dict:
            if method_kwargs.get('use_rle', True):
                # 使用RLE编码压缩
                rle_data = {
                    'frames': {},
                    'metadata': {
                        'num_frames': len(masks_dict),
                        'frame_indices': sorted(masks_dict.keys()),
                        'image_size': list(masks_dict[list(masks_dict.keys())[0]].shape)
                    }
                }
                
                for frame_idx, mask in masks_dict.items():
                    rle = encode_rle(mask)
                    rle_data['frames'][str(frame_idx)] = rle
                
                # 保存为JSON文件
                with open(compressed_file, 'w') as f:
                    json.dump(rle_data, f)
                
                # 计算压缩比
                original_size = sum(mask.nbytes for mask in masks_dict.values())
                compressed_size = compressed_file.stat().st_size
                compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
                print(f"  Saved compressed masks: {compressed_file}")
                print(f"  Compression ratio: {compression_ratio:.2f}x (original: {original_size/1024/1024:.2f}MB, compressed: {compressed_size/1024/1024:.2f}MB)")
            else:
                # 保存为单独的PNG文件（兼容模式）
                for frame_idx, mask in masks_dict.items():
                    mask_path = output_seq_dir / f"mask_{frame_idx:04d}.png"
                    cv2.imwrite(str(mask_path), mask * 255)


def main():
    parser = argparse.ArgumentParser(description="从光流文件生成运动区域的binary mask")
    parser.add_argument(
        "--input_root", "-i",
        required=True,
        help="输入根目录路径（包含flow输出的目录）"
    )
    parser.add_argument(
        "--output_root", "-o",
        required=True,
        help="输出根目录路径"
    )
    parser.add_argument(
        "--min_threshold",
        type=float,
        default=1.0,
        help="最小光流模长阈值（像素），大于此值的区域才会被考虑（默认：1.0）"
    )
    parser.add_argument(
        "--top_percentile",
        type=float,
        default=10.0,
        help="分位数（0-100），默认10（即10%分位数）"
    )
    parser.add_argument(
        "--noise_threshold",
        type=float,
        default=5.0,
        help="噪声阈值（像素），小于此值且不在前10%的区域会被去掉（默认：5.0）"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="如果输出文件已存在则跳过（默认强制覆盖）"
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="不保存可视化结果（默认保存）"
    )
    parser.add_argument(
        "--no_rle",
        action="store_true",
        help="不使用RLE压缩，保存为单独的PNG文件（默认使用RLE压缩）"
    )
    
    args = parser.parse_args()
    
    # 准备方法参数
    method_kwargs = {
        'save_vis': not args.no_vis,
        'use_rle': not args.no_rle,
        'min_threshold': args.min_threshold,
        'top_percentile': args.top_percentile,
        'noise_threshold': args.noise_threshold,
    }
    
    print(f"\nGenerating masks from flow files...")
    print(f"Input root: {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"Method: magnitude_threshold")
    print(f"Parameters:")
    print(f"  min_threshold: {args.min_threshold} pixels")
    print(f"  top_percentile: {args.top_percentile}% (using {args.top_percentile}% percentile as threshold)")
    print(f"  noise_threshold: {args.noise_threshold} pixels")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Save visualizations: {not args.no_vis}\n")
    
    process_data_directory(
        args.input_root, args.output_root, method='magnitude_threshold',
        skip_existing=args.skip_existing, **method_kwargs
    )
    
    print("\n所有处理完成！")


if __name__ == "__main__":
    main()
