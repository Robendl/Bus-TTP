import itertools
from typing import Dict

import xgboost as xgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm

from config.config import Config
from data.dataset_bundle import DatasetBundle


def merge_route_sample(
        x_df: pd.DataFrame,
        y_df: pd.Series,
        route_lookup: Dict[str, torch.Tensor],
        n_trips_per_route: int = 10,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    df = x_df.copy()
    print(df.shape)
    df["label"] = y_df
    df = (
        df.groupby("stop_to_stop_id", group_keys=False)
        .apply(lambda g: g.sample(
            n=min(len(g), n_trips_per_route),
            random_state=random_state
        ))
    )
    route_features = df["route_seq_hash"].astype(str).map(route_lookup)
    route_features = route_features.apply(lambda x: np.array(x).squeeze())
    route_features = pd.DataFrame(route_features.tolist(), index=df.index)

    x_out = pd.concat([df.drop(columns="label"), route_features], axis=1)
    y_out = df["label"]
    print(x_out.shape)
    return x_out, y_out


def xgboost_gridsearch(cfg: Config, db: DatasetBundle, route_lookup):
    X_train_with_ids, y_train = merge_route_sample(db.train.x, db.train.y, route_lookup, n_trips_per_route=2, random_state=cfg.training.random_state)
    X_train = X_train_with_ids.drop(["id", "route_seq_hash", "stop_to_stop_id"], axis=1)

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
            "device": "cuda",
            "eval_metric": "mae"
        })

        cv_results = xgb.cv(
            params=params,
            dtrain=dtrain,
            nfold=3,
            num_boost_round=2000,
            early_stopping_rounds=50,
            seed=cfg.training.random_state,
            verbose_eval=False
        )

        mean_mae = cv_results["test-mae-mean"].min()
        boost_rounds = cv_results["test-mae-mean"].idxmin()

        if mean_mae < best_score:
            best_score = mean_mae
            best_params = params
            best_num_boost = boost_rounds

    print("Best parameters:", best_params)
    print("Best MAE (CV):", best_score)
    print("Optimal n_estimators:", best_num_boost)


def fit_xgboost(cfg: Config, db: DatasetBundle, route_lookup):
    max_speed_index = cfg.dataset.route_feature_names.index('max_speed')

    X_train_with_ids = merge_route(db.train.x, route_lookup)
    X_train = X_train_with_ids.drop(["id", "route_seq_hash", "stop_to_stop_id"], axis=1)
    # X_val = merge_distance_max_speed(db.val.x, route_lookup, max_speed_index)
    X_test_with_ids = merge_route(db.test.x, route_lookup)
    X_test = X_test_with_ids.drop(["id", "route_seq_hash", "stop_to_stop_id"], axis=1)

    # Model initialiseren (regressie voorbeeld)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    final_model = xgb.train(params=best_params, dtrain=dtrain, num_boost_round=best_num_boost)
    dtest = xgb.DMatrix(X_test)
    y_pred = final_model.predict(dtest)

    # Evaluatie
    mae = mean_absolute_error(db.test.y, y_pred)
    print(f"MAE: {mae:.4f}")

