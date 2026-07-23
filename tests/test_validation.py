import pandas as pd

from src.validation import (
    apply_context_features,
    fit_stock_context,
    make_feature_columns,
    prepare_feature_frame,
)


def test_stock_context_is_fit_only_from_supplied_training_rows():
    train = pd.DataFrame(
        {
            "stock_id": [0, 0, 1, 1],
            "time_id": [1, 2, 1, 2],
            "rv1_full": [1.0, 3.0, 10.0, 14.0],
        }
    )
    context = fit_stock_context(train, ["rv1_full"])
    means = context.set_index("stock_id")["rv1_full_by_stock_mean"]
    assert means.loc[0] == 2.0
    assert means.loc[1] == 12.0


def test_validation_rows_receive_training_stock_context():
    train = pd.DataFrame(
        {
            "stock_id": [0, 0],
            "time_id": [1, 2],
            "rv1_full": [1.0, 3.0],
        }
    )
    valid = pd.DataFrame({"stock_id": [0], "time_id": [3], "rv1_full": [100.0]})
    context = fit_stock_context(train, ["rv1_full"])
    result = apply_context_features(valid, context, ["rv1_full"])
    assert result.loc[0, "rv1_full_by_stock_mean"] == 2.0
    assert result.loc[0, "rv1_full_minus_stock_mean"] == 98.0


def test_stock_id_can_be_excluded_for_ablation():
    frame = pd.DataFrame(
        {
            "stock_id": [0],
            "time_id": [1],
            "target": [0.1],
            "feature": [2.0],
        }
    )
    columns = make_feature_columns(frame, include_stock_id=False)
    assert columns == ["feature"]


def test_fill_values_are_reused_from_train():
    train = pd.DataFrame({"feature": [1.0, None, 3.0]})
    valid = pd.DataFrame({"feature": [None]})
    X_train, fill_values = prepare_feature_frame(train, ["feature"])
    X_valid, _ = prepare_feature_frame(valid, ["feature"], fill_values)
    assert X_train.loc[1, "feature"] == 2.0
    assert X_valid.loc[0, "feature"] == 2.0
