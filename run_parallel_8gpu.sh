#!/bin/bash

# 8-GPU parallel processing: each GPU processes one dataset at a time.
# When a GPU finishes, it automatically picks up the next unprocessed dataset.

DATA_ROOT="/home/unitree/remote_jensen2/Galaxea-Open-World-Dataset/lerobot_v3_0"
DATA_ROOT="/home/unitree/remote_jensen2/unitree_410_g1_rec"
OUTPUT_ROOT="/home/unitree/remote_jensen2/oxe_lerobot/unitree_410_g1_rec_lerobot_v3_0/optical_flow_annotations_waft"

WAFT_CFG="config/a1/tar-c-t.json"
WAFT_CKPT="checkpoints/tar-c-t.pth"

NUM_GPUS=8
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

# ── Collect all dataset names ──
DATASETS=()
for d in "$DATA_ROOT"/*; do
    [ -d "$d" ] && DATASETS+=("$(basename "$d")")
done

echo "=========================================="
echo "WAFT 8-GPU Parallel Processing"
echo "=========================================="
echo "Data root:   $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Datasets:    ${#DATASETS[@]}"
echo "GPUs:        $NUM_GPUS"
echo "=========================================="

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "No temp_* datasets found in $DATA_ROOT"
    exit 1
fi

# ── Write dataset names into a job queue file (one per line) ──
QUEUE_FILE=$(mktemp /tmp/waft_queue_XXXXXX)
for ds in "${DATASETS[@]}"; do
    echo "$ds"
done > "$QUEUE_FILE"

# Lock file for atomic job fetching
LOCK_FILE=$(mktemp /tmp/waft_lock_XXXXXX)

# ── Worker function: runs on one GPU, keeps pulling jobs until queue is empty ──
worker() {
    local gpu_id=$1

    while true; do
        # Atomically grab the next job from the queue
        local dataset
        dataset=$(
            flock "$LOCK_FILE" bash -c '
                line=$(head -n1 "'"$QUEUE_FILE"'")
                if [ -z "$line" ]; then
                    exit 1
                fi
                # Remove first line from queue
                tail -n +2 "'"$QUEUE_FILE"'" > "'"$QUEUE_FILE"'.tmp"
                mv "'"$QUEUE_FILE"'.tmp" "'"$QUEUE_FILE"'"
                echo "$line"
            '
        )

        # If no more jobs, exit
        if [ $? -ne 0 ] || [ -z "$dataset" ]; then
            echo "[GPU $gpu_id] No more datasets. Worker done."
            return 0
        fi

        echo "[GPU $gpu_id] Starting: $dataset"
        local log_file="$LOG_DIR/${dataset}_gpu${gpu_id}.log"

        CUDA_VISIBLE_DEVICES=$gpu_id python batch_annotate_flow_lerobot_waft.py \
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
            --max_vis_frames 100 \
            --skip_existing \
            --save_vis \
            --short_time_offset 0.3 \
            --short_min_threshold 1.0 \
            --no_bidirectional \
            --max-fps 5 \
            --datasets "$dataset" \
            --compensate_ego_motion \
            --ransac_threshold 3.0 \
            > "$log_file" 2>&1

        local status=$?
        if [ $status -eq 0 ]; then
            echo "[GPU $gpu_id] Finished: $dataset (success)"
        else
            echo "[GPU $gpu_id] Finished: $dataset (FAILED, exit=$status, see $log_file)"
        fi
    done
}

# ── Launch one worker per GPU ──
pids=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    worker "$gpu_id" &
    pids+=($!)
done

echo ""
echo "Launched ${#pids[@]} workers. PIDs: ${pids[*]}"
echo "Logs: $LOG_DIR/"
echo "Monitor: tail -f $LOG_DIR/*.log"
echo ""

# ── Wait for all workers to finish ──
all_ok=true
for pid in "${pids[@]}"; do
    wait "$pid" || all_ok=false
done

# Cleanup
rm -f "$QUEUE_FILE" "$LOCK_FILE"

echo ""
echo "=========================================="
if $all_ok; then
    echo "All datasets processed successfully!"
else
    echo "Some datasets failed. Check logs in: $LOG_DIR/"
fi
echo "Results: $OUTPUT_ROOT"
echo "=========================================="
