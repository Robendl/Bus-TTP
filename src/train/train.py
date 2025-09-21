import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config.config import Config, OptimizerConfig
from data.data_processing import create_dataloader
from model.lstm import LSTMFeedforwardCombination
from model.mlp import MLP
from plot.plot import plot_losses
from tqdm import tqdm
from train.eval import evaluate


def train_model(cfg: Config, model: MLP | LSTMFeedforwardCombination, train_loader, val_loader, optimCfg: OptimizerConfig, device, verbose=True):
    train_losses = []
    val_losses = []
    best_id_targets = []

    best_val_score = np.inf

    criterion = nn.SmoothL1Loss(beta=15.0)

    if optimCfg.type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=optimCfg.learning_rate, weight_decay=optimCfg.weight_decay)
    elif optimCfg.type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=optimCfg.learning_rate, weight_decay=optimCfg.weight_decay)
    else:
        raise Exception(f"Unknown optimizer {optimCfg.type}")

    if optimCfg.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                      patience=3, cooldown=0, min_lr=1e-6)
    elif optimCfg.scheduler == "None":
        scheduler = None
    else:
        raise Exception(f"Unknown scheduler {optimCfg.scheduler}")

    if verbose:
        print(f"Starting training {model.name}...", flush=True)
    epochs_without_improvement = 0

    validation_activated = False

    for epoch in range(cfg.training.epochs):
        model.train()
        running_loss = 0.0

        for _, x_batch, y_batch in tqdm(train_loader, disable=not verbose):
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            if model.name == "LSTM":
                time_features, padded_routes, lengths = x_batch
                time_features = time_features.to(device, non_blocking=True)
                padded_routes = padded_routes.to(device, non_blocking=True)
                predictions = model(time_features, padded_routes, lengths)
            else: # MLP
                x_batch = x_batch.to(device, non_blocking=True)
                predictions = model(x_batch)

            loss = criterion(predictions.view(-1), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        if verbose:
            print(f"Epoch {epoch + 1}/{cfg.training.epochs} - Loss: {avg_loss:.4f}", flush=True)
        train_losses.append(avg_loss)

        if avg_loss < 28 or epoch > 19:
            validation_activated = True

        if cfg.dataset.use_validation and ( not cfg.training.early_stopping_enabled or (cfg.training.early_stopping_enabled and validation_activated)):
            epochs_without_improvement += 1

            if epoch % cfg.training.eval_frequency == 0 or epoch == cfg.training.epochs - 1:
                (mae, _, _), _, _, id_targets, _ = evaluate(cfg, model, val_loader, device, verbose)
                val_losses.append(mae)
                if verbose:
                    print(f"Validation MAE: {mae:.3f}", flush=True)

                if mae < best_val_score:
                    if best_val_score - mae > cfg.training.min_delta:
                        epochs_without_improvement = 0
                    best_val_score = mae
                    best_id_targets = id_targets
                    output_dir = HydraConfig.get().run.dir
                    torch.save(model.state_dict(), f"{output_dir}/{model.name}.pth")

                if scheduler is not None:
                    scheduler.step(mae)

            if cfg.training.early_stopping_enabled and epochs_without_improvement >= cfg.training.patience:
                if verbose:
                    print("Early stopping", flush=True)
                break

        plot_losses(train_losses, val_losses, model.name)

    return train_losses, val_losses, best_id_targets, best_val_score
