from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def default_lgbm_params(seed=42):
    return dict(
        objective="regression",
        metric="l2",
        n_estimators=5000,
        learning_rate=0.015,
        num_leaves=96,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.30,
        reg_lambda=1.20,
        random_state=seed,
        n_jobs=-1,
        force_col_wise=True,
        verbosity=-1,
    )


def make_lgbm_regressor(params, seed):
    fold_params = params.copy()
    fold_params["random_state"] = seed
    return LGBMRegressor(**fold_params)


def compact_feature_set(feature_cols, feature_importance_summary, top_n=120):
    selected = []
    if "stock_id" in feature_cols:
        selected.append("stock_id")

    for feature in feature_importance_summary["feature"]:
        if feature in feature_cols and feature not in selected:
            selected.append(feature)
        if sum(col != "stock_id" for col in selected) >= top_n:
            break

    return selected


def make_catboost_params():
    return dict(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2500,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=4.0,
        random_strength=0.5,
        od_type="Iter",
        od_wait=150,
        thread_count=-1,
        allow_writing_files=False,
    )


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_mlp_baseline(feature_cols, seed=42):
    numeric_cols = [col for col in feature_cols if col != "stock_id"]
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if "stock_id" in feature_cols:
        transformers.append(("stock", make_one_hot_encoder(), ["stock_id"]))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return make_pipeline(
        preprocessor,
        MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=120,
            random_state=seed,
        ),
    )

