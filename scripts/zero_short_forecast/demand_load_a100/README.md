`zero_short_forecast/demand_load_a100/` contains A100-oriented zero-shot shell entrypoints for `dataset/demand_load/route_A/df_power.csv`.

Conventions:
- one `test` and one `forecast` script per model
- each model now has two explicit context variants:
  - `*_common_a100.sh`: common context for fair comparison
  - `*_optim_a100.sh`: model-specific optimized context
- `Moirai` and `TiRex` keep only `common` scripts because their optimized context is also `512`
- `test` scripts use fixed 8-GPU `DataParallel` with `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- `forecast` scripts default to a single GPU and can override it via `GPU_ID`
- scripts assume the server-side virtual environment is already activated and call `python -u run.py` directly

Defaults:
- target: `h_total_use`
- frequency: `5min`
- label length: `96`
- prediction length: `288`
- testing step: `1`

Common seq_len on A100:
- all models: `512`

Optimized seq_len on A100:
- `Chronos`: `2048`
- `Chronos2`: `4096`
- `Moirai`: `512`
- `Sundial_original_fix`: `2880`
- `TiRex`: `512`
- `TimeMoE_original_fix_50M`: `2048`
- `TimeMoE_original_fix_200M`: `2048`
- `TimesFM`: `4096`
