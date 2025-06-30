import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import omegaconf
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple

from config.config import Config
from plot.plot import plot_results
from tqdm import tqdm
from train.eval import evaluate, tolerance_accuracy_curve


def train_model(cfg: Config, model, train_loader, val_loader):

    # print("First eval")
    # targets, predictions, mse, mae = evaluate(model, val_loader)
    mse_list = []
    mae_list = []

    best_score = np.inf

    # Loss and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Training loop

    for epoch in range(cfg.training.epochs):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in tqdm(train_loader):
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions.view(-1), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{cfg.training.epochs} - Loss: {avg_loss:.4f}", flush=True)

        if (epoch + 1) % cfg.training.eval_frequency == 0 or epoch == cfg.training.epochs - 1:
            targets, predictions, mse, mae = evaluate(model, val_loader)
            mse_list.append(mse)
            mae_list.append(mae)
            # plot_results(mae_list, mse_list, baseline_mae, baseline_mse)
            print(f"Validation Results | MSE: {mse:.3f}, MAE: {mae:.3f}", flush=True)

            if mae < best_score:
                best_score = mae
                # tolerance_accuracy_curve(targets, predictions, y_pred_baseline, "Validation")
                output_dir = HydraConfig.get().run.dir
                torch.save(model.state_dict(), f"{output_dir}/lstm.pth")

    return model, mae_list, mse_list
