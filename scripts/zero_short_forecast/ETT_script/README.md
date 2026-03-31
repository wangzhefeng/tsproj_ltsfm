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

Each script keeps the original flat `python -u run.py ...` style and varies only by `model_name` and `seq_len`.
