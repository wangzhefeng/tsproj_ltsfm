Demand-load zero-shot scripts for `dataset/demand_load/route_A/df_power.csv`.

Current scripts:
- `Chronos2_df_power_test.sh`
- `Chronos2_df_power_predict.sh`
- `TimeMoE_original_fix_df_power_test.sh`
- `TimeMoE_original_fix_df_power_predict.sh`
- `Sundial_original_fix_df_power_test.sh`
- `Sundial_original_fix_df_power_predict.sh`

Defaults:
- target: `h_total_use`
- frequency: `5min`
- Chronos2 / TimeMoE context length: `2048`
- Sundial context length: `960`
- prediction length: `288`
