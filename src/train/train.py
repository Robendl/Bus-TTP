"""Training loop for the MLP and LSTM models."""
import time

import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config.config import Config, OptimizerConfig
from model.lstm import LSTMFeedforwardCombination
from model.mlp import MLP
from plot.plot import plot_losses
from train.eval import evaluate


# Validation tracking is suppressed during the early "warm-up" epochs to avoid
# acting on noisy losses. After this point, early stopping and checkpointing
# become active.
WARMUP_LOSS_THRESHOLD = 28.0
WARMUP_MIN_EPOCHS = 20


def _build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    if cfg.type == "Adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.type == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer type: {cfg.type}")


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: OptimizerConfig):
    if cfg.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, cooldown=0, min_lr=1e-6)
    if cfg.scheduler in ("None", None):
        return None
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


def _forward(model: nn.Module, x_batch, device: torch.device):
    if model.name == "LSTM":
        time_features, padded_routes, lengths = x_batch
        return model(
            time_features.to(device, non_blocking=True),
            padded_routes.to(device, non_blocking=True),
            lengths,
        )
    return model(x_batch.to(device, non_blocking=True))


def train_model(
    cfg: Config,
    model: MLP | LSTMFeedforwardCombination,
    train_loader,
    val_loader,
    optimizer_cfg: OptimizerConfig,
    device: torch.device,
    verbose: bool = True,
):
    output_dir = HydraConfig.get().run.dir
    optimizer = _build_optimizer(model, optimizer_cfg)
    scheduler = _build_scheduler(optimizer, optimizer_cfg)
    criterion = nn.SmoothL1Loss(beta=1.0)

    train_losses, val_losses = [], []
    batch_times, epoch_times = [], []
    best_val_score = np.inf
    best_id_targets = None
    epochs_without_improvement = 0
    validation_active = False

    if verbose:
        print(f"Starting training {model.name}", flush=True)

    for epoch in range(cfg.training.epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.perf_counter()

        for _, x_batch, y_batch in tqdm(train_loader, disable=not verbose):
            batch_start = time.perf_counter()
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            predictions = _forward(model, x_batch, device)
            loss = criterion(predictions.view(-1), y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_times.append(time.perf_counter() - batch_start)

        epoch_times.append(time.perf_counter() - epoch_start)
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        if verbose:
            print(f"Epoch {epoch + 1}/{cfg.training.epochs} - loss: {avg_loss:.4f}", flush=True)

        if avg_loss < WARMUP_LOSS_THRESHOLD or epoch >= WARMUP_MIN_EPOCHS:
            validation_active = True

        should_validate = (
            cfg.dataset.use_validation
            and (not cfg.training.early_stopping_enabled or validation_active)
        )
        if should_validate:
            epochs_without_improvement += 1
            is_eval_epoch = epoch % cfg.training.eval_frequency == 0 or epoch == cfg.training.epochs - 1
            if is_eval_epoch:
                (mae, _, _), _, _, id_targets, _, val_loss = evaluate(cfg, model, val_loader, device, verbose)
                val_losses.append(val_loss)
                if verbose:
                    print(f"Validation MAE: {mae:.3f}", flush=True)

                if mae < best_val_score:
                    if best_val_score - mae > cfg.training.min_delta:
                        epochs_without_improvement = 0
                    best_val_score = mae
                    best_id_targets = id_targets
                    torch.save(model.state_dict(), f"{output_dir}/{model.name}.pth")

                if scheduler is not None:
                    scheduler.step(mae)

            if cfg.training.early_stopping_enabled and epochs_without_improvement >= cfg.training.patience:
                if verbose:
                    print("Early stopping", flush=True)
                break

        plot_losses(train_losses, val_losses, model.name)

    if not cfg.dataset.use_validation:
        torch.save(model.state_dict(), f"{output_dir}/{model.name}.pth")

    return (
        train_losses,
        val_losses,
        best_id_targets,
        best_val_score,
        float(np.mean(epoch_times)),
        float(np.mean(batch_times)),
    )
