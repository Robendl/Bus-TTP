import itertools
from dataclasses import asdict
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
    y_df: pd.Series,
    route_lookup: Dict[str, torch.Tensor]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Voeg routefeatures toe aan x_df via route_lookup.
    """
    route_features = x_df["route_seq_hash"].astype(str).map(route_lookup)
    route_features = route_features.apply(lambda x: np.array(x).squeeze())
    route_features = pd.DataFrame(route_features.tolist(), index=x_df.index)

    x_out = pd.concat([x_df, route_features], axis=1)
    y_out = y_df  # labels ongewijzigd

    ids = x_out[["id"]]
    x_out.drop(["id", "route_seq_hash", "stop_to_stop_id"], axis=1, inplace=True)
    return x_out, y_out, ids


def xgboost_gridsearch(cfg: Config, db: DatasetBundle, route_lookup):
    X_sampled, y_sampled = sample_trips_per_route(db.train.x, db.train.y, n_trips_per_route=10, random_state=cfg.training.random_state)
    X_train, y_train, _ = merge_route_features(X_sampled, y_sampled, route_lookup)

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
            num_boost_round=2000,
            early_stopping_rounds=50,
            seed=cfg.training.random_state,
            verbose_eval=False
        )

        mean_mae = cv_results["test-mae-mean"].min()
        boost_rounds = cv_results["test-mae-mean"].argmin()

        print(boost_rounds, mean_mae, flush=True)

        if mean_mae < best_score:
            best_score = mean_mae
            best_params = params
            best_num_boost = boost_rounds

    print("Best parameters:", best_params)
    print("Best MAE (CV):", best_score)
    print("Optimal n_estimators:", best_num_boost)


def train_xgb(cfg: Config, db: DatasetBundle, route_lookup):
    X_sampled, y_sampled = sample_trips_per_route(db.train.x, db.train.y, n_trips_per_route=10,
                                                  random_state=cfg.training.random_state)
    X_train, y_train, _ = merge_route_features(X_sampled, y_sampled, route_lookup)
    print(X_train.shape)
    X_val, y_val, _ = merge_route_features(db.val.x, db.val.y, route_lookup)
    X_test, y_test, ids = merge_route_features(db.test.x, db.test.y, route_lookup)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

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
        num_boost_round=4000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
    )
    print(model.best_iteration, model.best_score)
    y_pred = model.predict(dtest)

    id_targets = pd.DataFrame({
        "id": ids["id"].values,
        "prediction": y_pred,
        "target": y_test
    })

    mae = mean_absolute_error(y_test, y_pred)
    print("XGBoost MAE:", mae)

    return id_targets

