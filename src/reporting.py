"""Validate tracked experiment evidence and build lightweight report assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResultSummary:
    baseline_rmspe: float
    mean_cv_rmspe: float
    oof_rmspe: float
    relative_reduction: float
    fold_count: int
    feature_count: int


def _normalize_svg(path: Path) -> None:
    """Remove generator timestamps/whitespace noise from a tracked SVG."""

    text = path.read_text(encoding="utf-8")
    normalized = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    path.write_text(normalized, encoding="utf-8")


def read_fold_scores(path: Path) -> pd.DataFrame:
    scores = pd.read_csv(path, dtype={"fold": str})
    required = {"fold", "rmspe", "n_train", "n_valid", "best_iteration"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"fold score report is missing columns: {missing}")

    fold_rows = scores[scores["fold"].str.fullmatch(r"\d+")].copy()
    if fold_rows.empty:
        raise ValueError("fold score report contains no numbered folds")
    if fold_rows["fold"].duplicated().any():
        raise ValueError("fold score report contains duplicate fold identifiers")
    if (fold_rows["rmspe"] <= 0).any():
        raise ValueError("fold RMSPE values must be positive")
    for column in ["n_train", "n_valid", "best_iteration"]:
        if fold_rows[column].isna().any() or (fold_rows[column] <= 0).any():
            raise ValueError(f"{column} must be positive for every numbered fold")

    labels = set(scores["fold"])
    if not {"mean", "oof"}.issubset(labels):
        raise ValueError("fold score report must contain mean and oof summary rows")

    stated_mean = float(scores.loc[scores["fold"] == "mean", "rmspe"].iloc[0])
    computed_mean = float(fold_rows["rmspe"].mean())
    if not np.isclose(stated_mean, computed_mean, atol=5e-5):
        raise ValueError(
            f"stated mean RMSPE {stated_mean:.5f} does not match folds {computed_mean:.5f}"
        )
    return scores


def read_feature_importance(path: Path) -> pd.DataFrame:
    importance = pd.read_csv(path)
    required = {"rank", "feature", "importance"}
    missing = sorted(required - set(importance.columns))
    if missing:
        raise ValueError(f"feature report is missing columns: {missing}")
    if importance.empty:
        raise ValueError("feature report is empty")
    if importance["rank"].duplicated().any() or importance["feature"].duplicated().any():
        raise ValueError("feature report contains duplicate ranks or features")
    if (importance["importance"] < 0).any():
        raise ValueError("feature importance values cannot be negative")
    return importance.sort_values("rank").reset_index(drop=True)


def build_result_summary(
    report_dir: Path,
    baseline_rmspe: float,
    feature_count: int,
) -> ResultSummary:
    scores = read_fold_scores(report_dir / "fold_scores.csv")
    fold_rows = scores[scores["fold"].str.fullmatch(r"\d+")]
    mean_cv = float(scores.loc[scores["fold"] == "mean", "rmspe"].iloc[0])
    oof = float(scores.loc[scores["fold"] == "oof", "rmspe"].iloc[0])
    if baseline_rmspe <= oof:
        raise ValueError("baseline RMSPE must exceed model OOF RMSPE for a positive reduction")
    return ResultSummary(
        baseline_rmspe=baseline_rmspe,
        mean_cv_rmspe=mean_cv,
        oof_rmspe=oof,
        relative_reduction=(baseline_rmspe - oof) / baseline_rmspe,
        fold_count=int(len(fold_rows)),
        feature_count=feature_count,
    )


def write_report_assets(report_dir: Path) -> list[Path]:
    """Regenerate SVG charts entirely from the tracked CSV evidence."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "optiver-rv"
    svg_metadata = {"Date": None}
    report_dir.mkdir(parents=True, exist_ok=True)
    scores = read_fold_scores(report_dir / "fold_scores.csv")
    importance = read_feature_importance(report_dir / "feature_importance_top20.csv")

    folds = scores[scores["fold"].str.fullmatch(r"\d+")].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(folds["fold"].astype(int), folds["rmspe"], marker="o", linewidth=2)
    ax.axhline(
        float(scores.loc[scores["fold"] == "oof", "rmspe"].iloc[0]),
        linestyle="--",
        linewidth=1.5,
        color="#B44C43",
        label="OOF RMSPE",
    )
    ax.set(title="Grouped cross-validation by time_id", xlabel="Fold", ylabel="RMSPE")
    ax.set_xticks(folds["fold"].astype(int))
    ax.legend(frameon=False)
    fig.tight_layout()
    fold_path = report_dir / "fold_scores.svg"
    fig.savefig(fold_path, format="svg", metadata=svg_metadata)
    plt.close(fig)
    _normalize_svg(fold_path)

    top = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    colors = ["#B44C43" if feature == "stock_id" else "#2878B5" for feature in top["feature"]]
    ax.barh(top["feature"], top["importance"], color=colors)
    ax.set(title="Top features by mean LightGBM gain", xlabel="Mean gain", ylabel="")
    fig.tight_layout()
    importance_path = report_dir / "feature_importance.svg"
    fig.savefig(importance_path, format="svg", metadata=svg_metadata)
    plt.close(fig)
    _normalize_svg(importance_path)
    return [fold_path, importance_path]
