import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from plot.plot import plot_tac, plot_error_histogram


def relative_tolerance_accuracy(targets, predictions, percentage):
    targets = np.array(targets)
    predictions = np.array(predictions)
    tolerance = (percentage / 100.0) * targets
    return np.mean(np.abs(targets - predictions) <= tolerance)

def tolerance_accuracy(targets, predictions, tolerance):
    targets = np.array(targets)
    predictions = np.array(predictions)
    errors = np.abs(targets - predictions)
    return np.mean(errors <= tolerance)

def tolerance_accuracy_curve(targets, predictions, y_pred_baseline, data_split):
    margins = np.arange(1, 61, 5)
    accuracies = [tolerance_accuracy(targets, predictions, tol) for tol in margins]
    base_accuracies = [tolerance_accuracy(targets, y_pred_baseline, tol) for tol in margins]
    plot_tac(margins, accuracies, base_accuracies, 's', data_split)

    percentages = np.arange(1, 41, 5)
    accuracies = [relative_tolerance_accuracy(targets, predictions, p) for p in percentages]
    base_accuracies = [relative_tolerance_accuracy(targets, y_pred_baseline, p) for p in percentages]
    plot_tac(percentages, accuracies, base_accuracies, 'p', data_split)

    return margins, accuracies

def test(model, test_loader, y_pred_baseline):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X
            batch_y = batch_y

            outputs = model(batch_X).squeeze()

            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())

    targets = np.array(targets).flatten()
    predictions = np.array(predictions).flatten()

    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)

    tolerance_accuracy_curve(targets, predictions, y_pred_baseline, "Test")
    errors = np.array(predictions) - np.array(targets)
    plot_error_histogram(errors)
    return mse, mae

def evaluate(model, val_loader):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_X, batch_y in tqdm(val_loader):
            batch_X = batch_X
            batch_y = batch_y

            outputs = model(batch_X)#.squeeze()

            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())

    targets = np.array(targets).flatten()
    predictions = np.array(predictions).flatten()

    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    return targets, predictions, mse, mae
