DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-bfloat16}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1024}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-96}"
STRIDE="${STRIDE:-96}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CHECKPOINT="${CHECKPOINT:-pretrain_models/TimeMoE-200M}"
OUTPUT_DIR="${OUTPUT_DIR:-results/time_moe/etth2_200m_a100}"

python models/time_moe_usage/run_benchmark.py \
  --data dataset/ETTh2.csv \
  --dataset-name ETTh2 \
  --target-col OT \
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
