# Optiver Realized Volatility Forecasting

This is my end-to-end realized volatility forecasting project based on the Optiver Kaggle dataset. I use the project to show how I approach a quantitative ML problem: start from market microstructure intuition, build features carefully, validate without leaking market events across folds, and compare the main model against sensible alternatives.

## Project Scope

- Feature engineering from limit-order-book and trade data.
- Weighted average prices, spreads, depth, imbalance, realized-volatility moments, tail-window features, and trade-flow features.
- Leakage-aware `GroupKFold` validation by `time_id`.
- Fold-safe stock-level priors and batch-level time context features.
- LightGBM main model trained on log-volatility and evaluated with RMSPE.
- Optional Optuna tuning plus model comparison against CatBoost and a tabular MLP baseline.
- Feature importance, benchmark comparison, residual diagnostics, and difficult-stock analysis.

## Main Artifact

- `optiver_realized_volatility_portfolio.ipynb`

## Run

On Kaggle, the default paths should work.

For local execution:

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

The notebook does not include the raw Kaggle data. Keep data files, feature caches, and submission files out of GitHub.

## Resume Bullets

- Engineered market microstructure features from high-frequency order book and trade data, including WAP returns, realized-volatility moments, spread/depth signals, order imbalance, recency windows, and cross-sectional context features.
- Built a leakage-aware LightGBM forecasting pipeline with `GroupKFold` by market time bucket, log-target modeling, RMSPE validation, feature importance analysis, and residual diagnostics.
- Added Optuna hyperparameter tuning and benchmarked LightGBM against CatBoost and a tabular neural baseline under the same grouped validation design.

## Dependencies

See `requirements.txt`.
