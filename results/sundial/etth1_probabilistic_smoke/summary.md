# Sundial benchmark summary

- dataset: `ETTh1`
- env: `local`
- checkpoint: `pretrain_models/sundial-base-128m`
- windows: `1`
- context_length: `128`
- prediction_length: `24`
- metrics: `{'mae': 1.4881319204966228, 'mse': 3.803201758115241, 'rmse': 1.9501799296770645, 'mape': 0.0511231138187569}`

- probabilistic_metrics: `{'pinball_loss_q10': 0.33999198913574213, 'pinball_loss_q50': 0.7106572190920512, 'pinball_loss_q90': 0.4947039270401004, 'interval_coverage_q05_q95': 0.625, 'interval_width_q05_q95': 4.015699648857115}`
- probabilistic_outputs: `sample_predictions.npz`, `quantiles.csv`, `probabilistic_metrics.json`, `probabilistic_plot.png`
