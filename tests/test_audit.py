import pandas as pd
import pytest

from src.audit import validate_book_frame, validate_feature_keys, validate_trade_frame


def valid_book_frame():
    return pd.DataFrame(
        {
            "time_id": [1, 1],
            "seconds_in_bucket": [0, 10],
            "bid_price1": [99.0, 100.0],
            "ask_price1": [101.0, 102.0],
            "bid_price2": [98.0, 99.0],
            "ask_price2": [102.0, 103.0],
            "bid_size1": [10, 11],
            "ask_size1": [12, 13],
            "bid_size2": [9, 10],
            "ask_size2": [14, 15],
        }
    )


def test_book_audit_accepts_valid_frame():
    summary = validate_book_frame(valid_book_frame())
    assert summary.rows == 2
    assert summary.time_ids == 1


def test_book_audit_rejects_crossed_quotes():
    frame = valid_book_frame()
    frame.loc[0, "bid_price1"] = 105.0
    with pytest.raises(ValueError, match="crossed"):
        validate_book_frame(frame)


def test_trade_audit_rejects_duplicate_second():
    frame = pd.DataFrame(
        {
            "time_id": [1, 1],
            "seconds_in_bucket": [10, 10],
            "price": [100.0, 100.1],
            "size": [5, 6],
            "order_count": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_trade_frame(frame)


def test_feature_key_audit_rejects_duplicate_stock_time():
    frame = pd.DataFrame({"stock_id": [0, 0], "time_id": [1, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_feature_keys(frame, "features")
