# Optiver Realized Volatility Forecasting

An end-to-end machine-learning project for the Kaggle Optiver Realized
Volatility Prediction dataset. The pipeline transforms limit-order-book and
trade events into stock/time-bucket features, predicts realized volatility with
LightGBM, and evaluates out-of-fold predictions with RMSPE.

## What this project demonstrates

- Market-microstructure feature engineering from book and trade events.
- Explicit data contracts for raw schemas, price/size ranges, and unique keys.
- Event-grouped cross-validation: all rows for one `time_id` stay in one fold.
- Fold-fitted stock context and missing-value imputation.
- Reusable Python modules, automated tests, and reproducible result charts.
- Transparent limitations around stock identity, validation design, and data
  availability.

## Verified experiment snapshot

The tracked result tables were exported from a completed run on the official
competition training data.

| Metric | Value |
|---|---:|
| Training rows | 428,932 |
| Engineered features | 283 |
| Simple realized-volatility baseline RMSPE | 0.49657 |
| Mean five-fold CV RMSPE | 0.23109 |
| LightGBM OOF RMSPE | 0.23119 |
| Relative RMSPE reduction vs. baseline | 53.44% |

![Fold-level RMSPE](reports/fold_scores.svg)

The five validation folds ranged from 0.22155 to 0.23927 RMSPE. The small spread
between the mean fold score and pooled OOF score is expected because fold sizes
are nearly equal.

![Feature importance](reports/feature_importance.svg)

`stock_id` has the highest mean gain. This is plausible because stocks have
persistent volatility levels and the competition test set uses the same stock
universe. It also means this experiment does **not** demonstrate generalization
to unseen stocks. A matched three-way sensitivity test—full model, no direct
`stock_id`, and no stock identity/history—is provided in
`scripts/run_stock_id_ablation.py`; no ablation result is claimed until the full
competition data is rerun.

See [`reports/RESULTS.md`](reports/RESULTS.md) for fold values, evidence scope,
and interpretation.

## Validation design

The primary experiment uses `GroupKFold` with `time_id` as the group:

1. A market `time_id` cannot appear in both training and validation.
2. Stock-level context statistics are calculated from each training fold only.
3. Missing-value medians are learned from each training fold and reused on its
   validation fold.
4. Predictions are transformed back from `log(target)` before RMSPE scoring.

This is an **event-grouped** evaluation, not a chronological backtest. It tests
unseen market time buckets for the same stock universe; it does not simulate
deployment through calendar time or performance on new stocks.

## Features

The feature set includes:

- WAP1/WAP2 log-return volatility and higher moments;
- quoted spreads, market depth, and order-book imbalance;
- trailing-window and 100-second-bin features;
- trade size, order count, price, VWAP, and realized-volatility summaries;
- cross-sectional context within a `time_id`;
- fold-fitted stock context and relative-to-context features.

The raw preprocessing code is in `src/features.py`; fold-safe context handling
is in `src/validation.py`.

## Repository structure

```text
.
├── optiver_realized_volatility_portfolio.ipynb
├── src/
│   ├── audit.py
│   ├── experiment.py
│   ├── features.py
│   ├── metrics.py
│   ├── model.py
│   ├── reporting.py
│   ├── utils.py
│   └── validation.py
├── scripts/
│   ├── build_report_assets.py
│   └── run_stock_id_ablation.py
├── tests/
├── reports/
│   ├── RESULTS.md
│   ├── PROVENANCE.md
│   ├── fold_scores.csv
│   ├── fold_scores.svg
│   ├── feature_importance_top20.csv
│   └── feature_importance.svg
├── .github/workflows/ci.yml
├── Makefile
├── requirements.txt
└── requirements-test.txt
```

## Reproduce the checks and charts

The repository does not redistribute Kaggle data. The tracked CSV evidence,
tests, and SVG reports can be validated without it:

```bash
python -m pip install -r requirements-test.txt
python -m pytest
PYTHONPATH=. python scripts/build_report_assets.py
```

The SVG generator is deterministic. CI rebuilds both charts and fails if they
drift from the tracked CSV tables.

## Run the full notebook

Download the competition dataset through Kaggle after accepting its rules, then
set the input and working directories:

```bash
python -m pip install -r requirements.txt
export OPTIVER_DATA_DIR=/path/to/optiver-realized-volatility-prediction
export OPTIVER_WORK_DIR=./working
jupyter notebook optiver_realized_volatility_portfolio.ipynb
```

For a reduced preprocessing run:

```bash
export DEBUG_STOCK_LIMIT=5
```

Optional, computationally expensive experiments:

```bash
export RUN_TUNING=1
export OPTUNA_TRIALS=15
export RUN_MODEL_COMPARISON=1
```

The notebook outputs are intentionally not committed. Durable evidence is
exported to `reports/`, while raw data, caches, and submissions remain ignored.

## Run the stock-ID ablation

After the notebook creates `train_base_features.parquet`:

```bash
PYTHONPATH=. python scripts/run_stock_id_ablation.py \
  --train-features working/train_base_features.parquet
```

This runs the same five `time_id`-grouped folds for the full model, without the
direct categorical `stock_id`, and without both direct identity and fold-fitted
stock history. It writes `reports/stock_id_ablation.csv`.

## Scope and limitations

- The result is a Kaggle supervised-learning experiment, not a trading strategy.
- `GroupKFold(time_id)` prevents event overlap but is not chronological
  validation.
- Transaction costs, latency, slippage, and live execution are outside scope.
- Gain importance measures model usage, not causal importance.
- The reported 53.44% reduction is relative to the defined simple baseline, not
  a public/private leaderboard comparison.

## Resume-ready description

- Engineered market-microstructure features from 428,932 order-book/trade
  samples, including WAP returns, spreads, depth, imbalance, trailing windows,
  and trade-flow summaries.
- Built a LightGBM forecasting pipeline with `time_id`-grouped validation,
  achieving 0.231 OOF RMSPE versus a 0.497 simple realized-volatility baseline.
