DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-bfloat16}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1024}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-96}"
STRIDE="${STRIDE:-256}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CHECKPOINT="${CHECKPOINT:-pretrain_models/TimeMoE-50M}"
OUTPUT_DIR="${OUTPUT_DIR:-results/time_moe/traffic_a100}"

python models/time_moe_usage/run_benchmark.py \
  --data dataset/traffic.csv \
  --dataset-name traffic \
  --target-col 0 \
  --context-length "$CONTEXT_LENGTH" \
  --prediction-length "$PREDICTION_LENGTH" \
  --stride "$STRIDE" \
  --sample-limit "$SAMPLE_LIMIT" \
  --batch-size "$BATCH_SIZE" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --device-map "$DEVICE_MAP" \
  --dtype "$DTYPE" \
  --env a100 \
  --output-dir "$OUTPUT_DIR" \
  --save-plot
