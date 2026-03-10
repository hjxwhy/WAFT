import sys
import argparse
import os
import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

from model import fetch_model
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt, coords_grid, bilinear_sampler

from scipy.interpolate import griddata

from dataloader.flow.chairs import FlyingChairs
from dataloader.flow.sintel import MpiSintel
from dataloader.flow.kitti import KITTI
from dataloader.flow.spring import Spring
from dataloader.stereo.tartanair import TartanAir

from inference_tools import InferenceWrapper, AverageMeter

# Optional: LeRobot PyAV decoder for fallback when OpenCV fails (e.g. AV1)
try:
    _lerobot_video_utils = None
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _lerobot_src = os.path.join(_repo_root, "lerobot", "src")
    if os.path.exists(_lerobot_src):
        sys.path.insert(0, _lerobot_src)
        from lerobot.datasets.video_utils import decode_video_frames_torchvision as _decode_video_frames_torchvision
        _lerobot_video_utils = True
except Exception:
    _lerobot_video_utils = False


def decode_video_frames_by_time(video_path, from_ts, to_ts, fps=30, backend="pyav"):
    """
    Decode video frames by time range. Tries PyAV (LeRobot) first, then OpenCV.
    Used as fallback when OpenCV direct read fails (e.g. AV1).
    Returns: (T, H, W, 3) numpy uint8 RGB.
    """
    expected_num_frames = int((to_ts - from_ts) * fps)
    timestamps = [float(from_ts + i / fps) for i in range(expected_num_frames)]
    try:
        if _lerobot_video_utils:
            frames_tensor = _decode_video_frames_torchvision(
                video_path=video_path,
                timestamps=timestamps,
                tolerance_s=1.0 / fps,
                backend=backend,
                log_loaded_timestamps=False,
            )
            frames = frames_tensor.permute(0, 2, 3, 1).numpy()
            if frames.dtype == np.float32 and frames.max() <= 1.0:
                frames = (frames * 255).astype(np.uint8)
            elif frames.dtype != np.uint8:
                frames = frames.astype(np.uint8)
        else:
            return _decode_video_frames_opencv(video_path, from_ts, to_ts, fps)
        if len(frames) == 0:
            raise ValueError(f"No frames decoded (ts: {from_ts}-{to_ts})")
        return frames
    except Exception as e:
        print(f"    Warning: PyAV decoding failed, falling back to OpenCV: {e}")
        return _decode_video_frames_opencv(video_path, from_ts, to_ts, fps)


def _decode_video_frames_opencv(video_path, from_ts, to_ts, fps=30):
    """Decode video by time range using OpenCV (frame index seek). Returns (T, H, W, 3) RGB uint8."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = fps
    start_frame_idx = int(from_ts * video_fps)
    end_frame_idx = int(to_ts * video_fps)
    num_frames = end_frame_idx - start_frame_idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"No frames decoded from {video_path} (ts: {from_ts}-{to_ts})")
    return np.array(frames)


def warp_with_flow(image, flow):
    N, _, H, W = image.shape
    coords2 = coords_grid(N, H, W, device=image.device).permute(0, 2, 3, 1)
    coords2 = coords2 + flow.permute(0, 2, 3, 1)
    warped_image = bilinear_sampler(image, coords2)
    return warped_image.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()

def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])

def vis_heatmap(name, image, heatmap):
    # theta = 0.01
    # print(heatmap.max(), heatmap.min(), heatmap.mean())
    heatmap = heatmap[:, :, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap = heatmap > 0.01
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = image * 0.3 + colored_heatmap * 0.7
    # Create a color bar
    height, width = image.shape[:2]
    color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)  # Adjust the height and colormap as needed
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    cv2.imwrite(name, overlay)

def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap

@torch.no_grad()
def demo_data(name, args, model, image1, image2, flow_gt, valid=None, tiling=False):
    path = f"demo/{name}/{args.name}/"
    os.system(f"mkdir -p {path}")
    H, W = image1.shape[2:]
    cv2.imwrite(f"{path}image1.jpg", cv2.cvtColor(image1[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{path}image2.jpg", cv2.cvtColor(image2[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    flow_gt_vis = flow_to_image(flow_gt[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"{path}gt.jpg", flow_gt_vis)
    output = model.calc_flow(image1, image2)
    for i in range(len(output['flow'])):
        flow= output['flow'][i]
        flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
        cv2.imwrite(f"{path}flow_{i}.jpg", flow_vis)
        diff = flow_gt - flow
        diff_vis = flow_to_image(diff[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
        cv2.imwrite(f"{path}error_{i}.jpg", diff_vis)
        if 'info' in output:
            info = output['info'][i]
            heatmap = get_heatmap(info, args)
            vis_heatmap(f"{path}heatmap_{i}.jpg", image1[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())
        if valid is None:
            N, _, H, W = flow.shape
            valid = torch.ones((N, H, W), device=flow.device)
        else:
            valid = valid.to(flow.device)
        epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
        epe = (epe * valid).sum() / valid.sum()
        print(f"EPE_step{i}: {epe.cpu().item()}")

@torch.no_grad()
def demo_chairs(model, args, device=torch.device('cuda')):
    dataset = FlyingChairs(split='training')
    image1, image2, flow_gt, _ = dataset[150]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('chairs', args, model, image1, image2, flow_gt)

def demo_sintel(model, args, device=torch.device('cuda')):
    dstype = 'final'
    dataset = MpiSintel(split='training', dstype=dstype)
    image1, image2, flow_gt, valid = dataset[400]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    valid = valid[None].to(device)
    demo_data('sintel', args, model, image1, image2, flow_gt, valid=valid, tiling=False)

def demo_kitti(model, args, device=torch.device('cuda')):
    dataset = KITTI(split='training')
    image1, image2, flow_gt, valid = dataset[100]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    valid = valid[None].to(device)
    demo_data('kitti', args, model, image1, image2, flow_gt, valid=valid, tiling=False)

@torch.no_grad()
def demo_spring(model, args, device=torch.device('cuda'), split='train'):
    dataset = Spring(split='val')
    idx = 175
    if split == 'train' or split == 'val':
        image1, image2, flow_gt, _ = dataset[idx]
    else:
        image1, image2,  _ = dataset[idx]
        h, w = image1.shape[1:]
        flow_gt = torch.zeros((2, h, w))

    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('spring', args, model, image1, image2, flow_gt, tiling=False)

@torch.no_grad()
def demo_tartanair(model, args, device=torch.device('cuda')):
    dataset = TartanAir()
    image1, image2, flow_gt, valid = dataset[289992]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('tartanair', args, model, image1, image2, flow_gt, valid=valid)

@torch.no_grad()
def demo_custom(model, args, device=torch.device('cuda')):
    image1 = cv2.imread('datasets/KITTI/2015/testing/image_2/000168_10.png')
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread('datasets/KITTI/2015/testing/image_2/000168_11.png')
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
    H, W = image1.shape[1:]
    flow_gt = torch.zeros([2, H, W], device=device)
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('custom_downsample', args, model, image1, image2, flow_gt)


def _read_frame_at(cap, frame_idx):
    """Read a single frame at given index. Returns None on failure."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return frame


def _save_flow_combined_vis(path, frame0, frame1, flow_vis_bgr, flow_np_hw2, k):
    """
    Save a combined visualization: Source | Target (1s later) | Flow (color) | Flow Magnitude.
    All panels resized to same height, then hstack with labels. No mask/covisibility.
    """
    target_h = frame0.shape[0]
    vis_images = [frame0, frame1, flow_vis_bgr]
    flow_magnitude = np.sqrt(flow_np_hw2[:, :, 0]**2 + flow_np_hw2[:, :, 1]**2)
    magnitude_normalized = (flow_magnitude / (flow_magnitude.max() + 1e-6) * 255).astype(np.uint8)
    magnitude_colormap = cv2.applyColorMap(magnitude_normalized, cv2.COLORMAP_JET)
    vis_images.append(magnitude_colormap)

    vis_resized = []
    for img in vis_images:
        h, w = img.shape[:2]
        if h != target_h:
            new_w = int(w * target_h / h)
            img = cv2.resize(img, (new_w, target_h))
        vis_resized.append(img)

    combined = np.hstack(vis_resized)
    labels = ['Source', 'Target (1s later)', 'Flow (color)', 'Flow Magnitude']
    label_y = 30
    label_x = 10
    for i, label in enumerate(labels):
        cv2.putText(combined, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        label_x += vis_resized[i].shape[1]

    cv2.imwrite(os.path.join(path, f"vis_combined_{k:03d}.jpg"), combined)


@torch.no_grad()
def demo_video(model, args, video_path, device=torch.device('cuda')):
    use_opencv = True
    fps = 30.0
    frame_interval = 30
    indices = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: OpenCV could not open video, using decode-by-time fallback: {video_path}")
        use_opencv = False
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
            print("Warning: FPS invalid or 0, using default 30")
        frame_interval = int(round(fps))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0:
            idx = 0
            while idx < total_frames:
                indices.append(idx)
                idx += frame_interval
        else:
            while True:
                frame_idx = len(indices) * frame_interval
                frame = _read_frame_at(cap, frame_idx)
                if frame is None:
                    break
                indices.append(frame_idx)

        if len(indices) >= 2:
            frame0 = _read_frame_at(cap, indices[0])
            frame1 = _read_frame_at(cap, indices[1])
            if frame0 is None or frame1 is None:
                print("Warning: OpenCV frame read failed (e.g. AV1), using decode-by-time fallback")
                use_opencv = False
        else:
            use_opencv = False

        if not use_opencv:
            cap.release()
            cap = None

    if use_opencv and len(indices) < 2:
        if cap is not None:
            cap.release()
        raise RuntimeError("Video has fewer than 2 sampled frames (need at least 2 for 1s-interval pairs). Try a longer video or check FPS.")

    basename = os.path.splitext(os.path.basename(video_path))[0]
    path = f"demo/video_{basename}/{args.name}/"
    os.makedirs(path, exist_ok=True)

    if use_opencv:
        for k in range(len(indices) - 1):
            i0, i1 = indices[k], indices[k + 1]
            frame0 = _read_frame_at(cap, i0)
            frame1 = _read_frame_at(cap, i1)
            if frame0 is None or frame1 is None:
                print(f"Skip pair ({i0}, {i1}): failed to read frame(s)")
                break
            image1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
            image2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
            image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)[None].to(device)

            output = model.calc_flow(image1, image2)
            flow = output['flow'][-1]
            flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
            flow_vis = flow_to_image(flow_np, convert_to_bgr=True)
            # cv2.imwrite(os.path.join(path, f"flow_{k:03d}.jpg"), flow_vis)
            # cv2.imwrite(os.path.join(path, f"image1_{k:03d}.jpg"), frame0)
            # cv2.imwrite(os.path.join(path, f"image2_{k:03d}.jpg"), frame1)
            _save_flow_combined_vis(path, frame0, frame1, flow_vis, flow_np, k)
            # if 'info' in output:
            #     info = output['info'][-1]
            #     heatmap = get_heatmap(info, args)
            #     vis_heatmap(
            #         os.path.join(path, f"heatmap_{k:03d}.jpg"),
            #         image1[0].permute(1, 2, 0).cpu().numpy(),
            #         heatmap[0].permute(1, 2, 0).cpu().numpy(),
            #     )
            print(f"Saved flow pair {k}: {i0} -> {i1} (t={k}s -> t={k+1}s)")
        cap.release()
    else:
        # Fallback: decode by time (PyAV then OpenCV), supports e.g. AV1 when OpenCV fails
        k = 0
        while True:
            try:
                frames = decode_video_frames_by_time(video_path, k, k + 2, fps=fps)
            except Exception as e:
                if k == 0:
                    raise RuntimeError(f"Could not decode video (OpenCV and decode-by-time failed): {e}")
                break
            if len(frames) < frame_interval + 1:
                if k == 0:
                    raise RuntimeError("Video has fewer than 2 sampled frames for 1s interval.")
                break
            frame0_rgb = frames[0]
            frame1_rgb = frames[frame_interval]
            frame0 = cv2.cvtColor(frame0_rgb, cv2.COLOR_RGB2BGR)
            frame1 = cv2.cvtColor(frame1_rgb, cv2.COLOR_RGB2BGR)
            image1 = torch.tensor(frame0_rgb, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
            image2 = torch.tensor(frame1_rgb, dtype=torch.float32).permute(2, 0, 1)[None].to(device)

            output = model.calc_flow(image1, image2)
            flow = output['flow'][-1]
            flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
            flow_vis = flow_to_image(flow_np, convert_to_bgr=True)
            # cv2.imwrite(os.path.join(path, f"flow_{k:03d}.jpg"), flow_vis)
            # cv2.imwrite(os.path.join(path, f"image1_{k:03d}.jpg"), frame0)
            # cv2.imwrite(os.path.join(path, f"image2_{k:03d}.jpg"), frame1)
            _save_flow_combined_vis(path, frame0, frame1, flow_vis, flow_np, k)
            # if 'info' in output:
            #     info = output['info'][-1]
            #     heatmap = get_heatmap(info, args)
            #     vis_heatmap(
            #         os.path.join(path, f"heatmap_{k:03d}.jpg"),
            #         frame0_rgb,
            #         heatmap[0].permute(1, 2, 0).cpu().numpy(),
            #     )
            print(f"Saved flow pair {k}: t={k}s -> t={k+1}s (decode-by-time)")
            k += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--ckpt', help='checkpoint path', required=True, type=str)
    parser.add_argument('--dataset', help='dataset to evaluate on', choices=['chairs', 'sintel', 'spring', 'tartanair', 'kitti'], default=None, type=str)
    parser.add_argument('--video', help='input video path (sample at 1s interval, estimate flow between consecutive samples)', default=None, type=str)
    parser.add_argument('--scale', help='scale factor for input images', default=0.0, type=float)
    args = parse_args(parser)
    if args.video is None and args.dataset is None:
        parser.error('either --dataset or --video is required')
    model = fetch_model(args)
    load_ckpt(model, args.ckpt)
    model = model.cuda()
    model.eval()
    wrapped_model = InferenceWrapper(model, scale=args.scale, train_size=args.image_size, pad_to_train_size=False, tiling=False)
    device = next(model.parameters()).device

    if args.video is not None:
        demo_video(wrapped_model, args, args.video, device=device)
        return
    if args.dataset == 'chairs':
        demo_chairs(wrapped_model, args, device)
    elif args.dataset == 'sintel':
        demo_sintel(wrapped_model, args, device)
    elif args.dataset == 'spring':
        demo_spring(wrapped_model, args, device)
    elif args.dataset == 'tartanair':
        demo_tartanair(wrapped_model, args, device)
    elif args.dataset == 'kitti':
        demo_kitti(wrapped_model, args, device)

if __name__ == '__main__':
    main()