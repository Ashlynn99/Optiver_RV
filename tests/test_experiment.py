import numpy as np
import pandas as pd

from src.experiment import run_grouped_lightgbm_cv


def synthetic_base_features():
    rows = []
    for time_id in range(12):
        market = 0.002 + 0.0001 * time_id
        for stock_id in range(3):
            rv = market * (1 + 0.2 * stock_id)
            rows.append(
                {
                    "stock_id": stock_id,
                    "time_id": time_id,
                    "rv1_full": rv,
                    "imbalance_mean_full": 0.1 * (stock_id - 1),
                    "target": rv * 1.05,
                }
            )
    return pd.DataFrame(rows)


def small_params():
    return {
        "objective": "regression",
        "metric": "l2",
        "n_estimators": 20,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "min_child_samples": 2,
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": -1,
    }


def test_grouped_experiment_produces_complete_oof_predictions():
    result = run_grouped_lightgbm_cv(
        synthetic_base_features(),
        n_splits=3,
        params=small_params(),
        early_stopping_rounds=5,
    )
    assert result.label == "full_model"
    assert len(result.fold_scores) == 3
    assert not np.isnan(result.oof_predictions).any()
    assert result.oof_rmspe > 0


def test_identity_free_variant_removes_stock_id_and_stock_context():
    result = run_grouped_lightgbm_cv(
        synthetic_base_features(),
        n_splits=3,
        include_stock_id=False,
        include_stock_context=False,
        params=small_params(),
        early_stopping_rounds=5,
    )
    assert result.label == "without_stock_identity_or_history"
    assert "stock_id" not in result.feature_columns
    assert not any("_by_stock_" in column for column in result.feature_columns)
