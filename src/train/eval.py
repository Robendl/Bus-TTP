"""Evaluation helpers for the neural models."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from torch import nn
from tqdm import tqdm

from config.config import Config


def relative_tolerance_accuracy(targets, predictions, percentage: float) -> float:
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    tolerance = (percentage / 100.0) * targets
    return float(np.mean(np.abs(targets - predictions) <= tolerance))


def tolerance_accuracy(targets, predictions, tolerance: float) -> float:
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    return float(np.mean(np.abs(targets - predictions) <= tolerance))


def compute_accuracies(cfg: Config, targets, predictions):
    margins = np.arange(1, cfg.plot.margins_max, cfg.plot.step_size)
    percentages = np.arange(1, cfg.plot.percentages_max, cfg.plot.step_size)
    abs_accuracies = [tolerance_accuracy(targets, predictions, m) for m in margins]
    rel_accuracies = [relative_tolerance_accuracy(targets, predictions, p) for p in percentages]
    return abs_accuracies, rel_accuracies


def evaluate(cfg: Config, model, loader, device, verbose: bool = True):
    model.eval()
    criterion = nn.SmoothL1Loss(beta=1.0)

    ids_list = []
    predictions = []
    targets = []
    total_loss = 0.0

    with torch.no_grad():
        for ids, x_batch, y_batch in tqdm(loader, disable=not verbose):
            y_batch = y_batch.to(device)
            if model.name == "LSTM":
                time_features, padded_routes, lengths = x_batch
                outputs = model(
                    time_features.to(device, non_blocking=True),
                    padded_routes.to(device, non_blocking=True),
                    lengths,
                )
            else:
                outputs = model(x_batch.to(device, non_blocking=True))

            total_loss += criterion(outputs.view(-1), y_batch).item()
            ids_list.extend(ids)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(y_batch.cpu().numpy())

    val_loss = total_loss / len(loader)
    targets = np.array(targets).flatten()
    predictions = np.array(predictions).flatten()

    abs_accuracies, rel_accuracies = compute_accuracies(cfg, targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    mape = mean_absolute_percentage_error(targets, predictions)
    rmse = root_mean_squared_error(targets, predictions)

    id_targets = pd.DataFrame({"id": ids_list, "prediction": predictions, "target": targets})
    raw_scores = {
        "MAE": mean_absolute_error(targets, predictions, multioutput="raw_values"),
        "MAPE": mean_absolute_percentage_error(targets, predictions, multioutput="raw_values"),
        "RMSE": root_mean_squared_error(targets, predictions, multioutput="raw_values"),
    }
    return (mae, mape, rmse), abs_accuracies, rel_accuracies, id_targets, raw_scores, val_loss
