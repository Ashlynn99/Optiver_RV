import re

import numpy as np
import pandas as pd

from .metrics import realized_absvar, realized_kurtosis, realized_skew, realized_volatility


TAIL_WINDOWS = {
    "full": 0,
    "tail_500": 100,
    "tail_400": 200,
    "tail_300": 300,
    "tail_200": 400,
    "tail_100": 500,
}


def extract_stock_id(file_path):
    match = re.search(r"stock_id=(\d+)", str(file_path))
    if match is None:
        raise ValueError(f"Cannot parse stock_id from path: {file_path}")
    return int(match.group(1))


def safe_divide(numerator, denominator, eps=1e-12):
    denominator = denominator.where(np.abs(denominator) > eps, np.nan)
    return numerator / denominator


def log_return(series):
    return np.log(series).diff()


def calc_row_slope(values):
    values = np.asarray(values, dtype=float)
    valid_idx = np.where(~np.isnan(values))[0]
    if len(valid_idx) < 2:
        return 0.0
    x = valid_idx.astype(float)
    y = values[valid_idx]
    if np.allclose(np.std(y), 0):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def calc_wap1(df):
    denom = df["bid_size1"] + df["ask_size1"]
    return safe_divide(
        df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"],
        denom,
    )


def calc_wap2(df):
    denom = df["bid_size2"] + df["ask_size2"]
    return safe_divide(
        df["bid_price2"] * df["ask_size2"] + df["ask_price2"] * df["bid_size2"],
        denom,
    )


def make_book_window_features(wdf, suffix):
    if wdf.empty:
        return pd.DataFrame(columns=["time_id"])

    grp = wdf.groupby("time_id", sort=False)
    ret1 = grp["log_return1"].agg(
        rv1=realized_volatility,
        absvar1=realized_absvar,
        rskew1=realized_skew,
        rkurt1=realized_kurtosis,
    )
    ret2 = grp["log_return2"].agg(
        rv2=realized_volatility,
        absvar2=realized_absvar,
        rskew2=realized_skew,
        rkurt2=realized_kurtosis,
    )
    other = grp.agg(
        spread1_mean=("price_spread1", "mean"),
        spread1_std=("price_spread1", "std"),
        spread1_max=("price_spread1", "max"),
        spread2_mean=("price_spread2", "mean"),
        spread2_std=("price_spread2", "std"),
        spread2_max=("price_spread2", "max"),
        bid_ask_spread_mean=("bid_ask_spread", "mean"),
        bid_ask_spread_std=("bid_ask_spread", "std"),
        wap_balance_mean=("wap_balance", "mean"),
        wap_balance_std=("wap_balance", "std"),
        total_volume_mean=("total_volume", "mean"),
        total_volume_std=("total_volume", "std"),
        total_volume_sum=("total_volume", "sum"),
        imbalance_mean=("volume_imbalance", "mean"),
        imbalance_std=("volume_imbalance", "std"),
        imbalance_abs_mean=("volume_imbalance_abs", "mean"),
        depth1_mean=("depth1", "mean"),
        depth2_mean=("depth2", "mean"),
        depth_ratio_mean=("depth_ratio", "mean"),
        book_pressure1_mean=("book_pressure1", "mean"),
        book_pressure1_std=("book_pressure1", "std"),
        tau_mean=("tau", "mean"),
        tau_std=("tau", "std"),
        n_updates=("seconds_in_bucket", "count"),
        wap1_mean=("wap1", "mean"),
        wap1_std=("wap1", "std"),
        wap1_last=("wap1", "last"),
        wap2_mean=("wap2", "mean"),
        wap2_std=("wap2", "std"),
        wap2_last=("wap2", "last"),
    )

    feat = ret1.join(ret2).join(other).reset_index()
    return feat.rename(columns={c: f"{c}_{suffix}" for c in feat.columns if c != "time_id"})


def make_book_bin_features(df):
    tmp = df[["time_id", "seconds_in_bucket", "log_return1"]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame(columns=["time_id"])

    tmp["bin_id"] = (tmp["seconds_in_bucket"] // 100).clip(0, 5).astype(int)
    bin_rv = (
        tmp.groupby(["time_id", "bin_id"])["log_return1"]
        .agg(realized_volatility)
        .unstack()
        .reindex(columns=list(range(6)))
    )
    bin_rv.columns = [f"rv1_bin_{i}" for i in bin_rv.columns]
    out = bin_rv.reset_index()

    bin_cols = [c for c in out.columns if c.startswith("rv1_bin_")]
    out["rv1_bin_mean"] = out[bin_cols].mean(axis=1)
    out["rv1_bin_std"] = out[bin_cols].std(axis=1)
    out["rv1_bin_slope"] = out[bin_cols].apply(calc_row_slope, axis=1)
    return out


def book_preprocessor(file_path):
    stock_id = extract_stock_id(file_path)
    df = pd.read_parquet(file_path)
    df = df.sort_values(["time_id", "seconds_in_bucket"]).reset_index(drop=True)

    df["wap1"] = calc_wap1(df)
    df["wap2"] = calc_wap2(df)
    df["log_return1"] = df.groupby("time_id")["wap1"].transform(log_return)
    df["log_return2"] = df.groupby("time_id")["wap2"].transform(log_return)

    mid1 = (df["ask_price1"] + df["bid_price1"]) / 2.0
    mid2 = (df["ask_price2"] + df["bid_price2"]) / 2.0
    df["price_spread1"] = safe_divide(df["ask_price1"] - df["bid_price1"], mid1)
    df["price_spread2"] = safe_divide(df["ask_price2"] - df["bid_price2"], mid2)
    df["bid_ask_spread"] = (
        (df["ask_price1"] - df["bid_price1"]) + (df["ask_price2"] - df["bid_price2"])
    ) / 2.0
    df["wap_balance"] = np.abs(df["wap1"] - df["wap2"])
    df["depth1"] = df["ask_size1"] + df["bid_size1"]
    df["depth2"] = df["ask_size2"] + df["bid_size2"]
    df["depth_ratio"] = safe_divide(df["depth1"], df["depth2"])
    df["total_volume"] = df["depth1"] + df["depth2"]
    df["volume_imbalance"] = safe_divide(
        df["bid_size1"] + df["bid_size2"] - df["ask_size1"] - df["ask_size2"],
        df["total_volume"],
    )
    df["volume_imbalance_abs"] = np.abs(df["volume_imbalance"])
    df["book_pressure1"] = safe_divide(
        df["bid_size1"] - df["ask_size1"],
        df["bid_size1"] + df["ask_size1"],
    )
    df["tau"] = df.groupby("time_id")["seconds_in_bucket"].diff().fillna(0)

    feat = None
    for suffix, lower_bound in TAIL_WINDOWS.items():
        cur = make_book_window_features(df[df["seconds_in_bucket"] >= lower_bound], suffix)
        feat = cur if feat is None else feat.merge(cur, on="time_id", how="outer")

    feat = feat.merge(make_book_bin_features(df), on="time_id", how="left")
    feat["stock_id"] = stock_id
    cols = ["stock_id", "time_id"] + [c for c in feat.columns if c not in ["stock_id", "time_id"]]
    return feat[cols]


def make_trade_window_features(wdf, suffix):
    if wdf.empty:
        return pd.DataFrame(columns=["time_id"])

    grp = wdf.groupby("time_id", sort=False)
    ret = grp["log_return"].agg(
        trade_rv=realized_volatility,
        trade_absvar=realized_absvar,
        trade_rskew=realized_skew,
        trade_rkurt=realized_kurtosis,
    )
    other = grp.agg(
        trade_size_mean=("size", "mean"),
        trade_size_std=("size", "std"),
        trade_size_sum=("size", "sum"),
        trade_size_max=("size", "max"),
        trade_order_count_mean=("order_count", "mean"),
        trade_order_count_std=("order_count", "std"),
        trade_order_count_sum=("order_count", "sum"),
        trade_order_count_max=("order_count", "max"),
        trade_price_mean=("price", "mean"),
        trade_price_std=("price", "std"),
        trade_n_prints=("seconds_in_bucket", "count"),
        trade_amount_sum=("amount", "sum"),
    )

    vwap = safe_divide(grp["amount"].sum(), grp["size"].sum()).rename("trade_vwap")
    feat = ret.join(other).join(vwap).reset_index()
    return feat.rename(columns={c: f"{c}_{suffix}" for c in feat.columns if c != "time_id"})


def trade_preprocessor(file_path):
    stock_id = extract_stock_id(file_path)
    df = pd.read_parquet(file_path)
    df = df.sort_values(["time_id", "seconds_in_bucket"]).reset_index(drop=True)

    df["log_return"] = df.groupby("time_id")["price"].transform(log_return)
    df["amount"] = df["price"] * df["size"]

    feat = None
    for suffix, lower_bound in TAIL_WINDOWS.items():
        cur = make_trade_window_features(df[df["seconds_in_bucket"] >= lower_bound], suffix)
        feat = cur if feat is None else feat.merge(cur, on="time_id", how="outer")

    feat["stock_id"] = stock_id
    cols = ["stock_id", "time_id"] + [c for c in feat.columns if c not in ["stock_id", "time_id"]]
    return feat[cols]

