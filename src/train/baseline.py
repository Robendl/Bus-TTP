from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from train.eval import tolerance_accuracy_curve


def get_baseline(X_train_scaled, y_train, X_test_scaled, y_test):
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    tolerance_accuracy_curve(y_test, y_pred, baseline=True)

    return mae, mse