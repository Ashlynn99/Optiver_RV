"""Reusable grouped-CV experiment runner for engineered Optiver features.

The official competition test set contains the same stock universe as training,
so ``stock_id`` is a legitimate categorical feature for the primary model.
Because it can absorb persistent stock-level volatility, this module also
supports a no-stock-id ablation and reports both variants on identical folds.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .metrics import rmspe
from .model import default_lgbm_params, make_lgbm_regressor
from .validation import (
    apply_context_features,
    context_seed_columns,
    fit_stock_context,
    make_feature_columns,
    prepare_feature_frame,
)


@dataclass
class CVResult:
    label: str
    fold_scores: pd.DataFrame
    oof_rmspe: float
    oof_predictions: np.ndarray
    feature_importance: pd.DataFrame
    feature_columns: list[str]


def _fold_matrices(
    base_df: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    seed_columns: list[str],
    include_stock_id: bool,
    include_stock_context: bool,
):
    train_base = base_df.iloc[train_idx].copy()
    valid_base = base_df.iloc[valid_idx].copy()

    # Stock aggregates and fill values are fit on the training fold only.
    stock_context = fit_stock_context(
        train_base,
        seed_columns if include_stock_context else [],
    )
    train_context = apply_context_features(train_base, stock_context, seed_columns)
    valid_context = apply_context_features(valid_base, stock_context, seed_columns)
    feature_columns = make_feature_columns(
        train_context,
        valid_context,
        include_stock_id=include_stock_id,
    )
    X_train, fill_values = prepare_feature_frame(train_context, feature_columns)
    X_valid, _ = prepare_feature_frame(valid_context, feature_columns, fill_values)
    return X_train, X_valid, feature_columns


def run_grouped_lightgbm_cv(
    base_df: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    include_stock_id: bool = True,
    include_stock_context: bool = True,
    params: dict | None = None,
    early_stopping_rounds: int = 250,
) -> CVResult:
    """Run event-grouped CV with fold-fitted context features.

    Grouping by ``time_id`` prevents the same market event from appearing in
    both train and validation. It is not a chronological or unseen-stock test.
    """

    from lightgbm import early_stopping

    required = {"stock_id", "time_id", "target"}
    missing = sorted(required - set(base_df.columns))
    if missing:
        raise ValueError(f"base feature frame is missing columns: {missing}")
    if base_df.duplicated(["stock_id", "time_id"]).any():
        raise ValueError("base feature frame contains duplicate stock_id/time_id rows")

    groups = base_df["time_id"].to_numpy()
    target = base_df["target"].to_numpy(dtype=float)
    log_target = np.log(np.maximum(target, 1e-12))
    seed_columns = context_seed_columns(base_df, base_df)
    splitter = GroupKFold(n_splits=n_splits)
    model_params = default_lgbm_params(seed) if params is None else params.copy()

    oof = np.full(len(base_df), np.nan)
    fold_rows: list[dict] = []
    importance_frames: list[pd.DataFrame] = []
    resolved_features: list[str] = []

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(base_df, log_target, groups),
        start=1,
    ):
        X_train, X_valid, resolved_features = _fold_matrices(
            base_df,
            train_idx,
            valid_idx,
            seed_columns,
            include_stock_id,
            include_stock_context,
        )
        model = make_lgbm_regressor(model_params, seed + fold)
        categorical = ["stock_id"] if include_stock_id else "auto"
        model.fit(
            X_train,
            log_target[train_idx],
            eval_X=X_valid,
            eval_y=log_target[valid_idx],
            eval_metric="l2",
            callbacks=[early_stopping(early_stopping_rounds, verbose=False)],
            categorical_feature=categorical,
        )
        best_iteration = model.best_iteration_ or model_params["n_estimators"]
        prediction = np.exp(model.predict(X_valid, num_iteration=best_iteration))
        prediction = np.clip(prediction, 1e-8, None)
        oof[valid_idx] = prediction
        fold_rows.append(
            {
                "fold": fold,
                "rmspe": rmspe(target[valid_idx], prediction),
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "best_iteration": best_iteration,
            }
        )
        importance_frames.append(
            pd.DataFrame(
                {
                    "feature": resolved_features,
                    "importance": model.booster_.feature_importance(
                        importance_type="gain"
                    ),
                    "fold": fold,
                }
            )
        )
        del X_train, X_valid, model
        gc.collect()

    if np.isnan(oof).any():
        raise RuntimeError("OOF predictions are incomplete")
    importance = (
        pd.concat(importance_frames, ignore_index=True)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    if include_stock_id and include_stock_context:
        label = "full_model"
    elif not include_stock_id and include_stock_context:
        label = "without_direct_stock_id"
    elif not include_stock_id and not include_stock_context:
        label = "without_stock_identity_or_history"
    else:
        label = "direct_stock_id_without_stock_history"
    return CVResult(
        label=label,
        fold_scores=pd.DataFrame(fold_rows),
        oof_rmspe=rmspe(target, oof),
        oof_predictions=oof,
        feature_importance=importance,
        feature_columns=resolved_features,
    )
