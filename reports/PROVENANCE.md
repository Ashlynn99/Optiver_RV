# Result Provenance

## Source

The CSV tables in this directory were exported from a completed local execution
of `optiver_realized_volatility_portfolio.ipynb` on the official Kaggle Optiver
Realized Volatility Prediction training data.

The repository does not include:

- Kaggle competition data;
- engineered parquet caches;
- serialized models;
- OOF row-level predictions;
- submission files.

These exclusions avoid redistributing competition data and keep the repository
small.

## Evidence levels

The repository separates three kinds of evidence:

1. **Stored experiment output:** `fold_scores.csv` and
   `feature_importance_top20.csv`.
2. **Derived presentation:** the SVG charts, regenerated deterministically from
   the two CSV tables.
3. **Executable integrity checks:** unit tests for metrics, raw-data contracts,
   feature calculations, fold-fitted context, and result-table consistency.

CI validates levels 2 and 3. It does not retrain the full model because the
competition data is not available in GitHub Actions.

## Claim policy

- Numerical claims in the README must be traceable to the tracked CSV tables.
- Rounded values are reported to five decimal places for RMSPE.
- The 53.44% reduction is calculated as
  `(0.49657 - 0.23119) / 0.49657`.
- The baseline is defined in `RESULTS.md`; it is not a leaderboard benchmark.
- No chronological, unseen-stock, live-trading, or causal claim is made.
- A stock-ID ablation result will be added only after the full-data experiment
  is actually run and exported.
