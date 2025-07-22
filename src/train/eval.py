import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from tqdm import tqdm

from config.config import Config
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

def compute_accuracies(cfg: Config, targets, predictions):
    margins = np.arange(1, cfg.plot.margins_max, cfg.plot.step_size)
    percentages = np.arange(1, cfg.plot.percentages_max, cfg.plot.step_size)

    abs_accuracies = [tolerance_accuracy(targets, predictions, tol) for tol in margins]
    relative_accuracies = [relative_tolerance_accuracy(targets, predictions, p) for p in percentages]
    return abs_accuracies, relative_accuracies

def evaluate(cfg, model, val_loader, device):
    model.eval()
    ids_list = []
    predictions = []
    targets = []

    with torch.no_grad():
        for ids, x_batch, y_batch in tqdm(val_loader):
            if model.name == "LSTM":
                time_features, padded_routes, lengths = x_batch
                time_features = time_features.to(device, non_blocking=True)
                padded_routes = padded_routes.to(device, non_blocking=True)
                x_batch = (time_features, padded_routes, lengths)
            elif model.name == "MLP":
                x_batch = x_batch.to(device, non_blocking=True)

            outputs = model(x_batch)#.squeeze()

            ids_list.extend(ids)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(y_batch.cpu().numpy())

    targets = np.array(targets).flatten()
    predictions = np.array(predictions).flatten()
    abs_accuracies, relative_accuracies = compute_accuracies(cfg, targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    id_targets = pd.DataFrame({"id": ids_list, "prediction": predictions, "target": targets})
    return mae, abs_accuracies, relative_accuracies, id_targets
