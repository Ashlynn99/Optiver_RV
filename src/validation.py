import numpy as np
import pandas as pd


FEATURE_EXCLUDE_COLS = {"row_id", "target", "time_id"}
MISSING_FLAG_EXCLUDE_COLS = {"row_id", "target", "stock_id", "time_id"}

CONTEXT_KEYWORDS = [
    "rv1_", "rv2_", "absvar", "rskew", "rkurt", "spread", "wap_balance",
    "total_volume", "imbalance", "depth", "pressure", "tau", "n_updates",
    "trade_rv", "trade_absvar", "trade_rskew", "trade_rkurt", "trade_size",
    "trade_order_count", "trade_price", "trade_amount", "trade_vwap",
    "trade_n_prints", "rv1_bin_",
]

DIFF_SEED_CANDIDATES = [
    "rv1_full", "rv2_full", "absvar1_full", "absvar2_full",
    "spread1_mean_full", "spread2_mean_full", "bid_ask_spread_mean_full",
    "wap_balance_mean_full", "total_volume_mean_full", "imbalance_mean_full",
    "depth1_mean_full", "depth2_mean_full", "book_pressure1_mean_full",
    "tau_mean_full", "trade_rv_full", "trade_absvar_full", "trade_rskew_full",
    "trade_size_mean_full", "trade_order_count_mean_full", "trade_price_mean_full",
    "trade_amount_sum_full", "trade_vwap_full", "rv1_bin_mean", "rv1_bin_slope",
]


def add_missing_flags(df):
    out = df.copy()
    feature_cols = [c for c in out.columns if c not in MISSING_FLAG_EXCLUDE_COLS]
    out["row_missing_count"] = out[feature_cols].isna().sum(axis=1).astype("int16")
    out["row_missing_ratio"] = out[feature_cols].isna().mean(axis=1).astype("float32")
    return out


def context_seed_columns(train_df, test_df):
    numeric_cols = [
        c for c in train_df.columns
        if c not in FEATURE_EXCLUDE_COLS
        and c in test_df.columns
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    return [c for c in numeric_cols if any(key in c for key in CONTEXT_KEYWORDS)]


def flatten_agg_columns(agg_df, key):
    scope = key.replace("_id", "")
    agg_df.columns = [
        key if col[0] == key else f"{col[0]}_by_{scope}_{col[1]}"
        for col in agg_df.columns
    ]
    return agg_df


def fit_stock_context(df, seed_cols):
    stock_ids = pd.DataFrame({"stock_id": sorted(df["stock_id"].unique())})
    if not seed_cols:
        return stock_ids
    agg = df.groupby("stock_id")[seed_cols].agg(["mean", "std"]).reset_index()
    return flatten_agg_columns(agg, "stock_id")


def add_time_context(df, seed_cols):
    out = df.copy()
    if not seed_cols:
        return out
    agg = out.groupby("time_id")[seed_cols].agg(["mean", "std"]).reset_index()
    agg = flatten_agg_columns(agg, "time_id")
    return out.merge(agg, on="time_id", how="left")


def add_relative_context_features(df, seed_cols):
    out = df.copy()
    for col in [c for c in DIFF_SEED_CANDIDATES if c in seed_cols]:
        stock_mean = f"{col}_by_stock_mean"
        time_mean = f"{col}_by_time_mean"
        if stock_mean in out.columns:
            out[f"{col}_minus_stock_mean"] = out[col] - out[stock_mean]
        if time_mean in out.columns:
            out[f"{col}_minus_time_mean"] = out[col] - out[time_mean]
    return out


def apply_context_features(base_df, stock_context, seed_cols):
    out = add_time_context(base_df, seed_cols)
    out = out.merge(stock_context, on="stock_id", how="left")
    out = add_relative_context_features(out, seed_cols)
    out = add_missing_flags(out)
    return out.loc[:, ~out.columns.duplicated()].copy()


def make_feature_columns(train_context, test_context=None):
    cols = [c for c in train_context.columns if c not in FEATURE_EXCLUDE_COLS]
    if test_context is not None:
        cols = [c for c in cols if c in test_context.columns]
    return cols


def prepare_feature_frame(df, feature_cols, fill_values=None):
    X = df.reindex(columns=feature_cols).copy()
    numeric_cols = [c for c in feature_cols if c != "stock_id"]
    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)

    if fill_values is None:
        fill_values = X[numeric_cols].median().fillna(0)

    X[numeric_cols] = X[numeric_cols].fillna(fill_values)
    if "stock_id" in X.columns:
        X["stock_id"] = X["stock_id"].astype("category")
    return X, fill_values

