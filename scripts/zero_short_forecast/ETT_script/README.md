`zero_short_forecast/ETT_script/` contains per-model zero-shot shell entrypoints for the ETTh1 dataset.

Current scripts:
- `Chronos_ETTh1.sh`
- `Chronos2_ETTh1.sh`
- `Moirai_ETTh1.sh`
- `Sundial_ETTh1.sh`
- `Sundial_benchmark_ETTh1.sh`
- `Sundial_original_fix_ETTh1.sh`
- `TiRex_ETTh1.sh`
- `TimeMoE_ETTh1.sh`
- `TimeMoE_benchmark_ETTh1.sh`
- `TimeMoE_original_fix_ETTh1.sh`
- `TimesFM_ETTh1.sh`

Additional context presets:
- `*_common.sh`: common fair-comparison context, fixed at `seq_len=512`
- `*_optim.sh`: model-specific recommended context

Stable 7-model family presets in this directory:
- `Chronos_ETTh1_common.sh` / `Chronos_ETTh1_optim.sh`
- `Chronos2_ETTh1_common.sh` / `Chronos2_ETTh1_optim.sh`
- `Moirai_ETTh1_common.sh` / `Moirai_ETTh1_optim.sh`
- `Sundial_original_fix_ETTh1_common.sh` / `Sundial_original_fix_ETTh1_optim.sh`
- `TiRex_ETTh1_common.sh` / `TiRex_ETTh1_optim.sh`
- `TimeMoE_original_fix_ETTh1_common.sh` / `TimeMoE_original_fix_ETTh1_optim.sh`
- `TimesFM_ETTh1_common.sh` / `TimesFM_ETTh1_optim.sh`

Forecast presets:
- `Chronos_ETTh1_forecast_common.sh` / `Chronos_ETTh1_forecast_optim.sh`
- `Chronos2_ETTh1_forecast_common.sh` / `Chronos2_ETTh1_forecast_optim.sh`
- `Moirai_ETTh1_forecast_common.sh` / `Moirai_ETTh1_forecast_optim.sh`
- `Sundial_original_fix_ETTh1_forecast_common.sh` / `Sundial_original_fix_ETTh1_forecast_optim.sh`
- `TiRex_ETTh1_forecast_common.sh` / `TiRex_ETTh1_forecast_optim.sh`
- `TimeMoE_original_fix_ETTh1_forecast_common.sh` / `TimeMoE_original_fix_ETTh1_forecast_optim.sh`
- `TimesFM_ETTh1_forecast_common.sh` / `TimesFM_ETTh1_forecast_optim.sh`

Each script keeps the original flat `python -u run.py ...` style and varies only by `model_name` and `seq_len`.
