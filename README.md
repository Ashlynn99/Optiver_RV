# Optiver Realized Volatility Forecasting

This is my end-to-end realized volatility forecasting project based on the Optiver Kaggle dataset. I use it to show how I approach a quantitative ML problem: start from market microstructure intuition, build features carefully, validate without leaking market events across folds, and compare the main model against sensible alternatives.

## Final Results

These metrics come from executed Optiver notebook output that was cross-checked across three local notebook copies. The repository does not include raw Kaggle data.

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

## Workflow

Text summary:

```text
Raw book/trade parquet
-> WAP, spread, depth, imbalance features
-> Tail-window aggregation
-> Fold-safe stock context + batch time context
-> GroupKFold by time_id
-> LightGBM log-target model
-> RMSPE, feature importance, residual diagnostics
```

```mermaid
flowchart TD
    A["Raw book/trade parquet"] --> B["WAP, spread, depth, imbalance features"]
    B --> C["Tail-window aggregation"]
    C --> D["Fold-safe stock context + batch time context"]
    D --> E["GroupKFold by time_id"]
    E --> F["LightGBM log-target model"]
    F --> G["RMSPE, feature importance, residual diagnostics"]
    F --> H["Optuna tuning and model comparison"]
```

## Project Scope

- Feature engineering from limit-order-book and trade data.
- Leakage-aware grouped validation by market time bucket.
- LightGBM main model trained on log-volatility and evaluated with RMSPE.
- Optional Optuna tuning plus model comparison against CatBoost and a tabular MLP baseline.
- Post-model diagnostics, feature importance analysis, and difficult-stock/time-bucket review.

## Repository Structure

```text
.
├── optiver_realized_volatility_portfolio.ipynb
├── src/
│   ├── features.py
│   ├── validation.py
│   ├── model.py
│   ├── metrics.py
│   └── utils.py
├── reports/
│   ├── model_summary.md
│   └── fold_scores.csv
├── requirements.txt
└── README.md
```

The notebook is the main report. The `src/` folder separates the reusable feature, validation, metric, and model utilities so the project is easier to inspect than a single long notebook.

## Report Outputs

The current report files include the verified validation numbers above. When the notebook is rerun with raw data, it also exports the diagnostic figures.

```text
reports/
  model_summary.md
  fold_scores.csv
  feature_importance_top20.csv
  feature_importance.png
  residual_diagnostics.png
```

## Run

On Kaggle, the default paths should work. For local execution:

```bash
export OPTIVER_DATA_DIR=/path/to/optiver-realized-volatility-prediction
export OPTIVER_WORK_DIR=./working
jupyter notebook optiver_realized_volatility_portfolio.ipynb
```

For a quick smoke test:

```bash
export DEBUG_STOCK_LIMIT=5
```

Optional experiments:

```bash
export RUN_TUNING=1
export OPTUNA_TRIALS=15
export RUN_MODEL_COMPARISON=1
```

## Limitation

This is a Kaggle-style supervised forecasting project. The validation design is leakage-aware by `time_id`, but the model is not a production trading strategy and does not simulate transaction costs, latency, slippage, or live execution.


## Dependencies

See `requirements.txt`.
