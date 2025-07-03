from typing import Dict

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from config.config import Config
from data.dataset_bundle import DatasetBundle
from plot.plot import plot_error_histogram
import config.paths as paths

def merge_distance_max_speed(df: pd.DataFrame, route_lookup, max_speed_index):
    result_df = df[["distance"]].copy()
    result_df["max_speed_val"] = df["route_seq_hash"].map(lambda h: route_lookup[str(h)][:, max_speed_index][0])
    return result_df

def linear_regression(cfg: Config, db: DatasetBundle, route_lookup):
    max_speed_index = cfg.training.route_feature_names.index('max_speed')

    X_train = merge_distance_max_speed(db.train.x, route_lookup, max_speed_index)
    X_val = merge_distance_max_speed(db.val.x, route_lookup, max_speed_index)
    X_test = merge_distance_max_speed(db.test.x, route_lookup, max_speed_index)

    model = LinearRegression()
    model.fit(X_train, db.train.y)
    val_y_pred = model.predict(X_val)
    val_mae = mean_absolute_error(db.val.y, val_y_pred)

    test_y_pred = model.predict(X_test)
    test_mae = mean_absolute_error(db.test.y, test_y_pred)

    errors = np.array(test_y_pred) - np.array(db.test.y)
    # plot_error_histogram(errors, baseline=True)

    return val_mae, val_y_pred, test_mae, test_y_pred