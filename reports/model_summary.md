# Model Summary

This report is based on executed Optiver notebook output that was cross-checked across three local notebook copies. The repository does not include raw Kaggle data.

## Final Results

- Baseline RMSPE: 0.49657
- LightGBM OOF RMSPE: 0.23119
- Mean CV RMSPE: 0.23109
- Relative RMSPE reduction: 53.44%
- Number of engineered features: 283
- Training rows: 428,932
- CV strategy: `GroupKFold` by `time_id`
- Main feature groups: WAP returns, realized-volatility moments, spread/depth signals, order imbalance, tail-window features, trade-flow features, and cross-sectional context features

## Fold Scores

| Fold | RMSPE | Train rows | Valid rows | Best iteration |
|---:|---:|---:|---:|---:|
| 1 | 0.22155 | 343,145 | 85,787 | 2,977 |
| 2 | 0.23927 | 343,145 | 85,787 | 3,392 |
| 3 | 0.23283 | 343,146 | 85,786 | 3,274 |
| 4 | 0.22484 | 343,146 | 85,786 | 2,551 |
| 5 | 0.23694 | 343,146 | 85,786 | 3,379 |

## Top Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `stock_id` | 35,408.2 |
| 2 | `rv1_bin_slope_by_time_mean` | 2,996.8 |
| 3 | `imbalance_std_full_by_time_mean` | 2,959.2 |
| 4 | `trade_order_count_mean_full_by_time_mean` | 2,773.6 |
| 5 | `imbalance_mean_full_by_time_std` | 2,720.8 |
| 6 | `imbalance_mean_full_by_time_mean` | 2,689.2 |
| 7 | `imbalance_std_full_by_time_std` | 2,651.8 |
| 8 | `n_updates_full_by_time_std` | 2,634.8 |
| 9 | `rskew1_full` | 2,407.0 |
| 10 | `rkurt1_full` | 2,385.4 |

## Report Files

- `reports/model_summary.md`
- `reports/fold_scores.csv`
- `reports/feature_importance_top20.csv`
- `reports/feature_importance.png` is generated when the notebook is rerun with raw data.
- `reports/residual_diagnostics.png` is generated when the notebook is rerun with raw data.

## Limitation

This is a Kaggle-style supervised forecasting project. The validation design is leakage-aware by `time_id`, but the model is not a production trading strategy and does not simulate transaction costs, latency, slippage, or live execution.
