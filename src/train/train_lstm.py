import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig

from config.config import Config
from model.lstm import LSTMFeedforwardCombination
from model.mlp import MLP
from plot.plot import plot_losses
from tqdm import tqdm
from train.eval import evaluate


def train_model(cfg: Config, model: MLP | LSTMFeedforwardCombination, train_loader, val_loader, device):
    # print("First eval")
    # targets, predictions, mae = evaluate(model, val_loader, device)
    train_losses = []
    val_losses = []

    best_val_score = np.inf

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    for epoch in range(cfg.training.epochs):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in tqdm(train_loader):
            if model.name == "LSTM":
                time_features, padded_routes, lengths = x_batch
                time_features = time_features.to(device, non_blocking=True)
                padded_routes = padded_routes.to(device, non_blocking=True)
                x_batch = (time_features, padded_routes, lengths)
            elif model.name == "MLP":
                x_batch = x_batch.to(device, non_blocking=True)

            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions.view(-1), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{cfg.training.epochs} - Loss: {avg_loss:.4f}", flush=True)
        train_losses.append(avg_loss)

        if epoch % cfg.training.eval_frequency == 0 or epoch == cfg.training.epochs - 1:
            mae, _, _ = evaluate(cfg, model, val_loader, device)
            val_losses.append(mae)
            plot_losses(train_losses, val_losses, model.name)
            print(f"Validation MAE: {mae:.3f}", flush=True)

            if mae < best_val_score:
                best_val_score = mae
                output_dir = HydraConfig.get().run.dir
                torch.save(model.state_dict(), f"{output_dir}/{model.name}.pth")

    return
