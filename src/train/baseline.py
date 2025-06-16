from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


def get_baseline(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test):
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    val_y_pred = model.predict(X_val_scaled)
    val_mse = mean_squared_error(y_val, val_y_pred)
    val_mae = mean_absolute_error(y_val, val_y_pred)

    test_y_pred = model.predict(X_test_scaled)
    test_mse = mean_squared_error(y_test, test_y_pred)
    test_mae = mean_absolute_error(y_test, test_y_pred)

    return val_mae, val_mse, val_y_pred, test_mae, test_mse, test_y_pred