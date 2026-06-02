# Model Summary

This report is generated from `optiver_realized_volatility_portfolio.ipynb` after the notebook is run with the Optiver data.

## Final Results

- Baseline RMSPE: pending notebook run
- LightGBM OOF RMSPE: pending notebook run
- Relative RMSPE reduction: pending notebook run
- Number of engineered features: pending notebook run
- CV strategy: `GroupKFold` by `time_id`
- Main feature groups: WAP returns, realized-volatility moments, spread/depth signals, order imbalance, tail-window features, trade-flow features, and cross-sectional context features

## Generated Report Files

- `reports/model_summary.md`
- `reports/fold_scores.csv`
- `reports/feature_importance.png`
- `reports/residual_diagnostics.png`

## Limitation

This is a Kaggle-style supervised forecasting project. The validation design is leakage-aware by `time_id`, but the model is not a production trading strategy and does not simulate transaction costs, latency, slippage, or live execution.

