DEVICE="${DEVICE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-auto}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-128}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-24}"
STRIDE="${STRIDE:-512}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
CHECKPOINT="${CHECKPOINT:-pretrain_models/sundial-base-128m}"
OUTPUT_DIR="${OUTPUT_DIR:-results/sundial/etth1_probabilistic_smoke}"
if [ -x ".venv/bin/python" ]; then
  DEFAULT_PYTHON=".venv/bin/python"
else
  DEFAULT_PYTHON="python3"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"

"$PYTHON_BIN" models/sundial_usage/run_benchmark.py \
  --data dataset/ETT-small/ETTh1.csv \
  --dataset-name ETTh1 \
  --target-col OT \
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
  --env local \
  --output-dir "$OUTPUT_DIR" \
  --save-plot
