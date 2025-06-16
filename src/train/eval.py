import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from sklearn.metrics import mean_squared_error, mean_absolute_error
import wandb
import numpy as np
import matplotlib.pyplot as plt

from plot.plot import plot_tac

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

def tolerance_accuracy_curve(targets, predictions, baseline=False):
    margins = np.arange(1, 61, 1)
    accuracies = [tolerance_accuracy(targets, predictions, tol) for tol in margins]
    # Plot
    plot_tac(margins, accuracies, 's', baseline)

    margins = np.arange(1, 61, 1)
    accuracies = [tolerance_accuracy(targets, predictions, tol) for tol in margins]
    # Plot
    plot_tac(margins, accuracies, 's', baseline)

    percentages = np.arange(1, 41)  # 1% t/m 20%
    accuracies = [relative_tolerance_accuracy(targets, predictions, p) for p in percentages]
    plot_tac(percentages, accuracies, 'p', baseline)

    return margins, accuracies


def evaluate(model, dataloader, best_score):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X
            batch_y = batch_y

            outputs = model(batch_X).squeeze()

            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())

    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    if mae < best_score:
        tolerance_accuracy_curve(targets, predictions)
    return mse, mae
