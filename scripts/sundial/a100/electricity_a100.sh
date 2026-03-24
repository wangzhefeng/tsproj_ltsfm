DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-bfloat16}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-2880}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-96}"
STRIDE="${STRIDE:-256}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
CHECKPOINT="${CHECKPOINT:-pretrain_models/sundial-base-128m}"
OUTPUT_DIR="${OUTPUT_DIR:-results/sundial/electricity_a100}"

python models/sundial_usage/run_benchmark.py \
  --data dataset/electricity.csv \
  --dataset-name electricity \
  --target-col 0 \
  --context-length "$CONTEXT_LENGTH" \
  --prediction-length "$PREDICTION_LENGTH" \
  --stride "$STRIDE" \
  --sample-limit "$SAMPLE_LIMIT" \
  --batch-size "$BATCH_SIZE" \
  --num-samples "$NUM_SAMPLES" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --device-map "$DEVICE_MAP" \
  --dtype "$DTYPE" \
  --env a100 \
  --output-dir "$OUTPUT_DIR" \
  --save-plot
