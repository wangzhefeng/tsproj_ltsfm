Demand-load zero-shot scripts for `dataset/demand_load/route_A/df_power.csv`.

Current scripts:
- `Chronos2_df_power_test.sh`
- `Chronos2_df_power_predict.sh`
- `Chronos2_df_power_test_common.sh`
- `Chronos2_df_power_test_optim.sh`
- `Chronos2_df_power_forecast_common.sh`
- `Chronos2_df_power_forecast_optim.sh`
- `Chronos_df_power_test.sh`
- `Chronos_df_power_predict.sh`
- `Chronos_df_power_test_common.sh`
- `Chronos_df_power_test_optim.sh`
- `Chronos_df_power_forecast_common.sh`
- `Chronos_df_power_forecast_optim.sh`
- `Moirai_df_power_test.sh`
- `Moirai_df_power_predict.sh`
- `Moirai_df_power_test_common.sh`
- `Moirai_df_power_test_optim.sh`
- `Moirai_df_power_forecast_common.sh`
- `Moirai_df_power_forecast_optim.sh`
- `TimeMoE_original_fix_50M_df_power_simple_test.sh`
- `TimeMoE_original_fix_50M_df_power_test_common.sh`
- `TimeMoE_original_fix_50M_df_power_test_optim.sh`
- `TimeMoE_original_fix_50M_df_power_forecast_common.sh`
- `TimeMoE_original_fix_50M_df_power_forecast_optim.sh`
- `TimeMoE_original_fix_200M_df_power_test_common.sh`
- `TimeMoE_original_fix_200M_df_power_test_optim.sh`
- `TimeMoE_original_fix_200M_df_power_forecast_common.sh`
- `TimeMoE_original_fix_200M_df_power_forecast_optim.sh`
- `Sundial_original_fix_df_power_test.sh`
- `Sundial_original_fix_df_power_predict.sh`
- `Sundial_original_fix_df_power_test_common.sh`
- `Sundial_original_fix_df_power_test_optim.sh`
- `Sundial_original_fix_df_power_forecast_common.sh`
- `Sundial_original_fix_df_power_forecast_optim.sh`
- `TimesFM_df_power_test.sh`
- `TimesFM_df_power_predict.sh`
- `TimesFM_df_power_test_common.sh`
- `TimesFM_df_power_test_optim.sh`
- `TimesFM_df_power_forecast_common.sh`
- `TimesFM_df_power_forecast_optim.sh`
- `TiRex_df_power_test.sh`
- `TiRex_df_power_predict.sh`
- `TiRex_df_power_test_common.sh`
- `TiRex_df_power_test_optim.sh`
- `TiRex_df_power_forecast_common.sh`
- `TiRex_df_power_forecast_optim.sh`

Defaults:
- target: `h_total_use`
- frequency: `5min`
- Chronos2 / TimeMoE context length: `2048`
- Sundial context length: `960`
- prediction length: `288`

Additional test presets:
- `*_common.sh`: fair-comparison context, fixed at `seq_len=512`
- `*_optim.sh`: model-specific recommended context

Additional forecast presets:
- `*_forecast_common.sh`: fair-comparison context, fixed at `seq_len=512`
- `*_forecast_optim.sh`: model-specific recommended context
