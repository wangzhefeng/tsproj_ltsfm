DEVICE="${DEVICE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-auto}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-128}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-24}"
STRIDE="${STRIDE:-4096}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CHECKPOINT="${CHECKPOINT:-pretrain_models/TimeMoE-50M}"
OUTPUT_DIR="${OUTPUT_DIR:-results/time_moe/electricity_auto_device_smoke}"

python models/time_moe_usage/run_benchmark.py \
  --data dataset/electricity.csv \
  --dataset-name electricity \
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
  --env local \
  --output-dir "$OUTPUT_DIR" \
  --save-plot
