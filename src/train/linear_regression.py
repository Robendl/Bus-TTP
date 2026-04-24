"""Two-feature linear-regression baseline (distance + max speed)."""
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

from config.config import Config
from data.dataset_bundle import DatasetBundle
from train.eval import compute_accuracies


def _merge_distance_max_speed(
    df: pd.DataFrame,
    route_lookup: Dict[str, torch.Tensor],
    max_speed_index: int,
) -> np.ndarray:
    distance = df[["distance"]].values.reshape(-1, 1)
    max_speed = np.array(
        [route_lookup[str(h)][0, max_speed_index].item() for h in df["route_seq_hash"]]
    ).reshape(-1, 1)
    return np.hstack([distance, max_speed])


def linear_regression(cfg: Config, db: DatasetBundle, route_lookup):
    max_speed_index = cfg.dataset.route_feature_names.index("max_speed")

    X_train = _merge_distance_max_speed(db.train.x, route_lookup, max_speed_index)
    X_test = _merge_distance_max_speed(db.test.x, route_lookup, max_speed_index)

    model = LinearRegression()
    model.fit(X_train, db.train.y)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(db.test.y, y_pred)
    mape = mean_absolute_percentage_error(db.test.y, y_pred)
    rmse = root_mean_squared_error(db.test.y, y_pred)

    id_targets = pd.DataFrame({
        "id": db.test.x["id"],
        "prediction": y_pred,
        "target": db.test.y,
    })
    abs_accuracies, rel_accuracies = compute_accuracies(cfg, db.test.y, y_pred)

    return 0, (mae, mape, rmse), abs_accuracies, rel_accuracies, id_targets
