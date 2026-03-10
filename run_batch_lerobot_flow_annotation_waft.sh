#!/bin/bash

# 批量处理所有数据集的光流标注（使用WAFT模型）
# 处理所有temp_*数据集，每个数据集的cam_high和cam_side（如果可用）

DATA_ROOT="/home/unitree/remote_jensen2/oxe_lerobot/robochallenge_lerobot_v3_0/robochallenge_all_temp"
DATA_ROOT="/home/unitree/remote_jensen2/Galaxea-Open-World-Dataset/lerobot_v3_0"
DATA_ROOT="/home/unitree/remote_jensen2/unitree_401_g1/"
DATA_ROOT="/home/unitree/remote_jensen2/oxe_lerobot/oxe_lerobot_v3_0/"
OUTPUT_ROOT="/home/unitree/remote_jensen2/oxe_lerobot/oxe_lerobot_v3_0/optical_flow_annotations_waft"

# WAFT模型配置
WAFT_CFG="config/a1/tar-c-t.json"
WAFT_CKPT="checkpoints/tar-c-t.pth"  # 请修改为实际的checkpoint路径

echo "=========================================="
echo "WAFT Batch Optical Flow Annotation"
echo "=========================================="
echo "Data root: $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "WAFT Config: $WAFT_CFG"
echo "WAFT Checkpoint: $WAFT_CKPT"
echo "Time offset: 1.0s (30 frames)"
echo "Cameras: cam_high, cam_side"
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_ROOT"

# 运行批处理
CUDA_VISIBLE_DEVICES=7 python batch_annotate_flow_lerobot_waft.py \
    --data_root "$DATA_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --cfg "$WAFT_CFG" \
    --ckpt "$WAFT_CKPT" \
    --scale 0.0 \
    --time_offset 1 \
    --min_threshold 3.0 \
    --top_percentile 10 \
    --noise_threshold 10.0 \
    --device cuda \
    --max_vis_frames 500 \
    --skip_existing \
    --datasets fold_towel \
    --short_time_offset 0.3 \
    --short_min_threshold 1.0 \
    --no_bidirectional \
    --max-fps 5 \
    --datasets bridge_1.0.0_lerobot


    # --save_vis \
    # --datasets temp_arrange_paper_cups
    #     --compensate_ego_motion \
    # --ransac_threshold 3.0 \

echo ""
echo "=========================================="
echo "Batch processing completed!"
echo "Results saved to: $OUTPUT_ROOT"
echo "Check processing_summary.json for details"
echo "=========================================="
