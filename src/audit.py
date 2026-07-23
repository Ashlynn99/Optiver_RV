"""Data-contract checks for the Optiver book and trade inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


BOOK_REQUIRED_COLUMNS = {
    "time_id",
    "seconds_in_bucket",
    "bid_price1",
    "ask_price1",
    "bid_price2",
    "ask_price2",
    "bid_size1",
    "ask_size1",
    "bid_size2",
    "ask_size2",
}

TRADE_REQUIRED_COLUMNS = {
    "time_id",
    "seconds_in_bucket",
    "price",
    "size",
    "order_count",
}


@dataclass(frozen=True)
class AuditSummary:
    """Compact, serializable summary of one validated table."""

    name: str
    rows: int
    time_ids: int
    min_second: int
    max_second: int


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _require_no_duplicate_events(df: pd.DataFrame, name: str) -> None:
    keys = ["time_id", "seconds_in_bucket"]
    duplicate_count = int(df.duplicated(keys).sum())
    if duplicate_count:
        raise ValueError(
            f"{name} contains {duplicate_count} duplicate time_id/seconds_in_bucket rows"
        )


def _require_second_range(df: pd.DataFrame, name: str) -> None:
    seconds = df["seconds_in_bucket"]
    invalid = seconds.isna() | ~seconds.between(0, 599)
    if invalid.any():
        raise ValueError(f"{name} has {int(invalid.sum())} rows outside second range [0, 599]")


def _require_finite_positive(df: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values) | (values <= 0)
        if invalid.any():
            raise ValueError(
                f"{name}.{column} has {int(invalid.sum())} non-finite or non-positive values"
            )


def _require_finite_nonnegative(df: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values) | (values < 0)
        if invalid.any():
            raise ValueError(
                f"{name}.{column} has {int(invalid.sum())} non-finite or negative values"
            )


def _summary(df: pd.DataFrame, name: str) -> AuditSummary:
    return AuditSummary(
        name=name,
        rows=int(len(df)),
        time_ids=int(df["time_id"].nunique()),
        min_second=int(df["seconds_in_bucket"].min()),
        max_second=int(df["seconds_in_bucket"].max()),
    )


def validate_book_frame(df: pd.DataFrame, name: str = "book") -> AuditSummary:
    """Validate the raw schema and basic market-data invariants for one stock."""

    _require_columns(df, BOOK_REQUIRED_COLUMNS, name)
    if df.empty:
        raise ValueError(f"{name} is empty")
    _require_no_duplicate_events(df, name)
    _require_second_range(df, name)
    _require_finite_positive(
        df,
        ["bid_price1", "ask_price1", "bid_price2", "ask_price2"],
        name,
    )
    _require_finite_nonnegative(
        df,
        ["bid_size1", "ask_size1", "bid_size2", "ask_size2"],
        name,
    )
    crossed = (df["bid_price1"] > df["ask_price1"]) | (
        df["bid_price2"] > df["ask_price2"]
    )
    if crossed.any():
        raise ValueError(f"{name} has {int(crossed.sum())} crossed order-book rows")
    return _summary(df, name)


def validate_trade_frame(df: pd.DataFrame, name: str = "trade") -> AuditSummary:
    """Validate the raw schema and basic trade-data invariants for one stock."""

    _require_columns(df, TRADE_REQUIRED_COLUMNS, name)
    if df.empty:
        raise ValueError(f"{name} is empty")
    _require_no_duplicate_events(df, name)
    _require_second_range(df, name)
    _require_finite_positive(df, ["price", "size"], name)
    _require_finite_nonnegative(df, ["order_count"], name)
    return _summary(df, name)


def validate_feature_keys(df: pd.DataFrame, name: str) -> None:
    """Require one engineered row per stock and time bucket."""

    required = {"stock_id", "time_id"}
    _require_columns(df, required, name)
    duplicate_count = int(df.duplicated(["stock_id", "time_id"]).sum())
    if duplicate_count:
        raise ValueError(f"{name} has {duplicate_count} duplicate stock_id/time_id rows")
