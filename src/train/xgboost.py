import itertools
import os
from dataclasses import asdict
from typing import Dict

import xgboost as xgb
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm

from config.config import Config
from data.dataset_bundle import DatasetBundle


def sample_trips_per_route(
    x_df: pd.DataFrame,
    y_df: pd.Series,
    n_trips_per_route,
    random_state,
) -> tuple[pd.Series, pd.Series]:
    df = x_df.copy()
    df["label"] = y_df

    df = (
        df.groupby("stop_to_stop_id", group_keys=False)
          .apply(lambda g: g.sample(
              n=min(len(g), n_trips_per_route),
              random_state=random_state
          ))
    )

    x_out = df.drop(columns=["label"])
    y_out = df["label"]
    return x_out, y_out


def merge_route_features(
    x_df: pd.Series,
    route_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = x_df.merge(
        route_df,
        how="left",
        on="route_seq_hash",
        suffixes=("", "_route"),
    )

    ids = merged[["id"]]
    merged.drop(["id", "route_seq_hash", "stop_to_stop_id"], axis=1, inplace=True)
    return merged, ids


def xgboost_gridsearch(cfg: Config, db: DatasetBundle, route_df: pd.DataFrame):
    X_sampled, y_train = sample_trips_per_route(db.train.x, db.train.y, n_trips_per_route=10, random_state=cfg.training.random_state)
    X_train, _ = merge_route_features(X_sampled, route_df)

    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Hyperparam grid
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_lambda": [1, 5, 10],
        "reg_alpha": [0, 0.1, 1]
    }

    best_score = float("inf")
    best_params = None
    best_num_boost = None

    total_iterations = 1
    for value in param_grid.values():
        total_iterations *= len(value)


    device = "cpu" if cfg.dataset.use_subset else "cuda"
    keys, values = zip(*param_grid.items())
    for combo in tqdm(itertools.product(*values), total=total_iterations, disable=False):
        params = dict(zip(keys, combo))
        params.update({
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": device,
            "eval_metric": "mae"
        })

        cv_results = xgb.cv(
            params=params,
            dtrain=dtrain,
            nfold=3,
            metrics="mae",
            num_boost_round=2000,
            early_stopping_rounds=50,
            seed=cfg.training.random_state,
            maximize=False,
            verbose_eval=False
        )

        mean_mae = cv_results["test-mae-mean"].min()
        boost_rounds = cv_results["test-mae-mean"].argmin()
        print(boost_rounds, mean_mae, cv_results["test-mae-mean"].idxmin(), flush=True)

        if mean_mae < best_score:
            best_score = mean_mae
            best_params = params
            best_num_boost = boost_rounds

    print("Best parameters:", best_params)
    print("Best MAE (CV):", best_score)
    print("Optimal n_estimators:", best_num_boost)


def train_xgb(cfg: Config, db: DatasetBundle, route_df: pd.DataFrame, output_dir):
    # X_sampled, y_sampled = sample_trips_per_route(db.train.x, db.train.y, n_trips_per_route=1000,
    #                                               random_state=cfg.training.random_state)
    y_train = db.train.y
    X_train, _ = merge_route_features(db.train.x, route_df)
    print(X_train.shape)
    y_test = db.test.y
    X_test, ids = merge_route_features(db.test.x, route_df)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_test.columns.tolist())

    if cfg.dataset.use_validation:
        y_val = db.val.y
        X_val, _ = merge_route_features(db.val.x, route_df)
        y_test = db.test.y
        X_test, ids = merge_route_features(db.test.x, route_df)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_val.columns.tolist())
        evals = [(dtrain, "train"), (dval, "val")]
        early_stopping_rounds = 50
        num_boost_round = 2000
    else:
        evals = [(dtrain, "train")]
        early_stopping_rounds = None
        num_boost_round = 130

    device = "cpu" if cfg.dataset.use_subset else "cuda"

    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": device,
        "eval_metric": "mae",
        "max_depth": cfg.model.xgboost.max_depth,
        "learning_rate": cfg.model.xgboost.learning_rate,
        "subsample": cfg.model.xgboost.subsample,
        "colsample_bytree": cfg.model.xgboost.colsample_bytree,
        "reg_lambda": cfg.model.xgboost.reg_lambda,
        "reg_alpha": cfg.model.xgboost.reg_alpha,
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    if cfg.dataset.use_validation:
        print("Best iteration:", model.best_iteration)
        print("Best score:", model.best_score, flush=True)
    model.save_model(output_dir + "/xgboost.json")

    y_pred = model.predict(dtest)

    id_targets = pd.DataFrame({
        "id": ids["id"].values,
        "prediction": y_pred,
        "target": y_test
    })

    mae = mean_absolute_error(y_test, y_pred)
    print("XGBoost MAE:", mae, flush=True)

    # -- SHAP:

    n_samples = 100_000
    idx = np.random.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)

    X_test_sub = X_test.iloc[idx]

    X_test_sub.columns = [
        col
        .replace("on_road_", "")
        .replace("avg", "mean")
        .replace("perc", "")
        .replace("_", " ")
        .capitalize()
        for col in X_test.columns
    ]


    explainer = shap.TreeExplainer(model, X_test_sub)
    shap_values_full = explainer(X_test_sub)

    # barplot (global)
    ax = shap.plots.bar(shap_values_full, max_display=30, show=False)
    plt.tight_layout()
    ax.figure.savefig(f"{output_dir}/shap_bar.pdf")
    plt.clf()
    plt.close()

    # beeswarm (global + value direction)
    ax = shap.plots.beeswarm(shap_values_full, max_display=30, show=False)
    plt.tight_layout()
    ax.figure.savefig(f"{output_dir}/shap_beeswarm.pdf")
    plt.clf()
    plt.close()

    return id_targets

