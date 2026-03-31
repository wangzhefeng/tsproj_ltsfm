#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-0}"
MODEL_NAME="${MODEL_NAME:-Chronos2}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-10}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.matplotlib}"

mkdir -p "$MPLCONFIGDIR"
export MPLCONFIGDIR

if [[ "$DEVICE" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
  GPU_ARGS=(--use_gpu --gpu_type cuda --gpu 0)
elif [[ "$DEVICE" == "mps" ]]; then
  GPU_ARGS=(--use_gpu --gpu_type mps)
else
  GPU_ARGS=(--no_use_gpu)
fi

run_case() {
  local data_name="$1"
  local data_file="$2"
  local d_model="$3"
  local pred_len="$4"

  "$PYTHON_BIN" -u run.py \
    --task_name zero_shot_forecast \
    --is_training 0 \
    --root_path ./dataset/ETT-small/ \
    --data_path "$data_file" \
    --model_id "${data_name}_${SEQ_LEN}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$SEQ_LEN" \
    --pred_len "$pred_len" \
    --seg_len 24 \
    --enc_in 7 \
    --d_model "$d_model" \
    --dropout 0.5 \
    --learning_rate 0.0001 \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --des Exp \
    --itr 1 \
    "${GPU_ARGS[@]}"
}

for pred_len in 96 192 336 720; do
  run_case ETTh1 ETTh1.csv 512 "$pred_len"
done

for pred_len in 96 192 336 720; do
  run_case ETTh2 ETTh2.csv 256 "$pred_len"
done

for pred_len in 192 336 720; do
  run_case ETTm1 ETTm1.csv 512 "$pred_len"
done

for pred_len in 96 192 336 720; do
  run_case ETTm2 ETTm2.csv 512 "$pred_len"
done
