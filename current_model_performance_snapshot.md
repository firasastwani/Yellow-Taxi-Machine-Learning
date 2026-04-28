# Current Model Performance Snapshot

Date: 2026-04-26
Notebook: `data/yellow_taxi_data_science_workflow.ipynb`
Purpose: Baseline snapshot before adding Jan 2024 for cross-year evaluation.

## Primary Aggregate Benchmark (latest run)


| Model                         | Scope     | MAE     | MAPE     | n    |
| ----------------------------- | --------- | ------- | -------- | ---- |
| XGBoost                       | macro_avg | 29.2395 | 135.1421 | 904  |
| TimeMCL (median point policy) | macro_avg | 39.8650 | 134.4836 | 2712 |
| GBRT_fallback                 | micro_all | 30.2863 | 135.1421 | 904  |
| TimeMCL (median point policy) | micro_all | 39.8650 | 134.4836 | 2712 |


Notes:

- `GBRT_fallback` is used because `xgboost` could not load (`libomp.dylib` missing).
- `n` differs because the latest notebook state included duplicate TimeMCL rows in `pred_frames` during iterative reruns.

## TimeMCL Probabilistic Diagnostics (latest run)

Macro values:

- pinball_q10: 10.2515
- pinball_q50: 17.7922
- pinball_q90: 10.6430
- coverage_10_90: 0.5077
- mean_width_10_90: 62.0063

## Other Existing Baselines Still in Notebook

From earlier benchmark cell outputs (same notebook):

- Linear Regression (test): MAE = 4.337, RMSE = 14.337
- Lag baselines (test MAE):
  - Lag-1h: 5.982
  - Lag-24h: 5.108
  - Lag-168h: 4.486

## Reproducibility Reminder

Before final month-over-month comparison, rerun benchmark cells from a clean kernel so all models are evaluated with consistent `n` and no accumulated `pred_frames` state.