from typing import Dict

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from config.config import Config
from data.dataset_bundle import DatasetBundle
from plot.plot import plot_error_histogram
import config.paths as paths


def get_baseline(cfg: Config, db: DatasetBundle):
    route_lookup = np.load(paths.DATASETS_DIR + cfg.dataset.route_aggr + '.npz')
    index = cfg.dataset.route_aggr.index('max_speed')
    X_train = db.train.x["route_seq_hash"].map(lambda h: route_lookup[str(h)].shape[index])
    # df["route_feat_mean_0"] = df["route_seq_hash"].map(lambda h: route_lookup[str(h)][:, 0].mean())

    model = LinearRegression()
    model.fit(db.train.x, db.train.y)
    val_y_pred = model.predict(db.val.x)
    val_mse = mean_squared_error(db.val.y, val_y_pred)
    val_mae = mean_absolute_error(db.val.y, val_y_pred)

    test_y_pred = model.predict(db.test.x)
    test_mse = mean_squared_error(db.test.y, test_y_pred)
    test_mae = mean_absolute_error(db.test.y, test_y_pred)

    errors = np.array(test_y_pred) - np.array(db.test.y)
    plot_error_histogram(errors, baseline=True)

    return val_mae, val_mse, val_y_pred, test_mae, test_mse, test_y_pred