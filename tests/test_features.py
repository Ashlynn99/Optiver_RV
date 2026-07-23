import numpy as np
import pandas as pd
import pytest

from src.features import calc_wap1, extract_stock_id, log_return, safe_divide


def test_extract_stock_id():
    assert extract_stock_id("/data/book/stock_id=42") == 42
    with pytest.raises(ValueError):
        extract_stock_id("/data/book/unknown")


def test_safe_divide_masks_zero_denominator():
    numerator = pd.Series([4.0, 4.0])
    denominator = pd.Series([2.0, 0.0])
    result = safe_divide(numerator, denominator)
    assert result.iloc[0] == 2.0
    assert np.isnan(result.iloc[1])


def test_wap_uses_opposite_side_sizes():
    frame = pd.DataFrame(
        {
            "bid_price1": [99.0],
            "ask_price1": [101.0],
            "bid_size1": [30.0],
            "ask_size1": [10.0],
        }
    )
    # A larger bid size moves WAP toward the ask.
    assert calc_wap1(frame).iloc[0] == pytest.approx(100.5)


def test_log_return_is_bucket_local_when_grouped():
    frame = pd.DataFrame(
        {"time_id": [1, 1, 2, 2], "wap": [100.0, 101.0, 200.0, 202.0]}
    )
    result = frame.groupby("time_id")["wap"].transform(log_return)
    assert result.isna().sum() == 2
    assert result.iloc[1] == pytest.approx(np.log(1.01))
    assert result.iloc[3] == pytest.approx(np.log(1.01))
