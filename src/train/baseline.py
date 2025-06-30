from typing import Dict

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from plot.plot import plot_error_histogram


def get_baseline(data_splits: Dict[str, pd.DataFrame]):
    model = LinearRegression()
    model.fit(data_splits['X_train'], data_splits['y_train'])
    val_y_pred = model.predict(data_splits['X_val'])
    val_mse = mean_squared_error(data_splits['y_val'], val_y_pred)
    val_mae = mean_absolute_error(data_splits['y_val'], val_y_pred)

    test_y_pred = model.predict(X_test_scaled)
    test_mse = mean_squared_error(y_test, test_y_pred)
    test_mae = mean_absolute_error(y_test, test_y_pred)

    errors = np.array(test_y_pred) - np.array(y_test)
    plot_error_histogram(errors, baseline=True)

    return val_mae, val_mse, val_y_pred, test_mae, test_mse, test_y_pred