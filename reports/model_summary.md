# Model Summary

This short file is retained for compatibility with the original notebook
export. The canonical result interpretation is in [RESULTS.md](RESULTS.md), and
the evidence history is in [PROVENANCE.md](PROVENANCE.md).

## Result snapshot

- Baseline RMSPE: 0.49657
- LightGBM OOF RMSPE: 0.23119
- Mean five-fold RMSPE: 0.23109
- Relative RMSPE reduction: 53.44%
- Training rows: 428,932
- Engineered features: 283
- Validation: `GroupKFold(n_splits=5)` grouped by `time_id`

These are stored experiment outputs, not values regenerated in CI. CI checks
the result tables for internal consistency and deterministically rebuilds the
tracked SVG charts.

## Interpretation boundary

Grouping by `time_id` prevents the same market event from appearing in both
train and validation. It is not chronological validation and does not test
unseen stocks. `stock_id` is the top gain-importance feature, so no claim of
cross-stock generalization is made.

## Tracked evidence

- `fold_scores.csv`
- `feature_importance_top20.csv`
- `fold_scores.svg`
- `feature_importance.svg`
