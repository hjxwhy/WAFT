#!/bin/bash

# Optimized batch optical flow annotation (WAFT v3)
# 3-stage async pipeline: prefetch / batched GPU / postprocess

DATA_ROOT="/home/unitree/remote_jensen2/oxe_lerobot/oxe_lerobot_v3_0/"
OUTPUT_ROOT="/home/unitree/remote_jensen2/oxe_lerobot/oxe_lerobot_v3_0/optical_flow_annotations_waft"

WAFT_CFG="config/a1/tar-c-t.json"
WAFT_CKPT="checkpoints/tar-c-t.pth"

# Prefetch: number of processes for parallel video decoding
NUM_PREFETCH_WORKERS=8
# Prefetch: max episodes pre-decoded in memory
PREFETCH_BUFFER=16
# Postprocess: number of processes for mask generation + RLE save
NUM_POSTPROCESS_WORKERS=8
# Max flow pairs per calc_flow call (tune to fit GPU memory)
MAX_FLOW_BATCH=16

echo "=========================================="
echo "WAFT Batch Optical Flow Annotation (v3)"
echo "=========================================="
echo "Data root: $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "WAFT Config: $WAFT_CFG"
echo "Prefetch workers: $NUM_PREFETCH_WORKERS (buffer=$PREFETCH_BUFFER)"
echo "Postprocess workers: $NUM_POSTPROCESS_WORKERS"
echo "Max flow batch: $MAX_FLOW_BATCH"
echo "=========================================="
echo ""

mkdir -p "$OUTPUT_ROOT"

CUDA_VISIBLE_DEVICES=7 python batch_annotate_flow_lerobot_waft_v3.py \
    --data_root "$DATA_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --cfg "$WAFT_CFG" \
    --ckpt "$WAFT_CKPT" \
    --scale 0.0 \
    --time_offset 1 \
    --min_threshold 3.0 \
    --top_percentile 10 \
    --noise_threshold 10.0 \
    --max_vis_frames 500 \
    --skip_existing \
    --short_time_offset 0.3 \
    --short_min_threshold 1.0 \
    --no_bidirectional \
    --max-fps 5 \
    --num_prefetch_workers "$NUM_PREFETCH_WORKERS" \
    --prefetch_buffer "$PREFETCH_BUFFER" \
    --num_postprocess_workers "$NUM_POSTPROCESS_WORKERS" \
    --max_flow_batch "$MAX_FLOW_BATCH" \
    --datasets bridge_1.0.0_lerobot

    # --fp16 \
    # --save_vis \
    # --compensate_ego_motion \
    # --ransac_threshold 3.0 \

echo ""
echo "=========================================="
echo "Batch processing completed!"
echo "Results saved to: $OUTPUT_ROOT"
echo "Check processing_summary.json for details"
echo "=========================================="
