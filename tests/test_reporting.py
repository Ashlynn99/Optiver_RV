from pathlib import Path

import pandas as pd
import pytest

from src.reporting import build_result_summary, read_feature_importance, read_fold_scores


REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def test_tracked_fold_report_is_internally_consistent():
    scores = read_fold_scores(REPORT_DIR / "fold_scores.csv")
    assert len(scores[scores["fold"].str.fullmatch(r"\d+")]) == 5


def test_tracked_feature_report_is_valid():
    importance = read_feature_importance(REPORT_DIR / "feature_importance_top20.csv")
    assert importance.iloc[0]["feature"] == "stock_id"
    assert len(importance) == 20


def test_result_summary_matches_tracked_claim():
    summary = build_result_summary(REPORT_DIR, baseline_rmspe=0.49657, feature_count=283)
    assert summary.oof_rmspe == pytest.approx(0.23119)
    assert summary.relative_reduction == pytest.approx(0.534426, abs=1e-6)


def test_fold_report_rejects_incorrect_mean(tmp_path):
    bad = pd.DataFrame(
        [
            {"fold": "1", "rmspe": 0.2, "n_train": 10, "n_valid": 5, "best_iteration": 4},
            {"fold": "2", "rmspe": 0.3, "n_train": 10, "n_valid": 5, "best_iteration": 4},
            {"fold": "mean", "rmspe": 0.9},
            {"fold": "oof", "rmspe": 0.25},
        ]
    )
    path = tmp_path / "fold_scores.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not match"):
        read_fold_scores(path)
