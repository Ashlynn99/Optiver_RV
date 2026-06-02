import gc
import glob

import pandas as pd


def env_flag(name, default=False):
    import os

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def env_int(name, default=None):
    import os

    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def assert_data_available(config):
    required_paths = [
        config.train_csv_path,
        config.test_csv_path,
        config.book_train_dir,
        config.book_test_dir,
        config.trade_train_dir,
        config.trade_test_dir,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Competition data was not found. Set OPTIVER_DATA_DIR to the Optiver data folder. "
            f"Missing paths: {missing}"
        )


def list_stock_partitions(path, limit=None):
    files = sorted(glob.glob(str(path / "stock_id=*")))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No stock partitions found under {path}")
    return files


def run_preprocessor(file_list, preprocessor, label):
    feat_list = []
    total = len(file_list)

    for i, file_path in enumerate(file_list, start=1):
        print(f"{label}: {i}/{total}", end="\r")
        feat_list.append(preprocessor(file_path))
        if i % 10 == 0:
            gc.collect()

    print(f"{label}: done ({total}/{total})")
    if not feat_list:
        return pd.DataFrame(columns=["stock_id", "time_id"])
    return pd.concat(feat_list, ignore_index=True)


def merge_feature_blocks(base_df, *blocks):
    out = base_df.copy()
    for block in blocks:
        if block.empty:
            continue
        out = out.merge(block, on=["stock_id", "time_id"], how="left")
    return out


def cache_is_usable(train_df, test_df):
    only_train = sorted(set(train_df.columns) - set(test_df.columns) - {"target"})
    only_test = sorted(set(test_df.columns) - set(train_df.columns) - {"row_id"})
    return len(only_train) == 0 and len(only_test) == 0

