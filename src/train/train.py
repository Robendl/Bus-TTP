import torch
import torch.nn as nn
import wandb
from omegaconf import omegaconf
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple

from config.config import Config
from plot.plot import plot_results
from train.eval import evaluate


def train_model(cfg: Config, model, train_loader, eval_loader, baseline_mae, baseline_mse):
    # Initialize Weights & Biases
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project=cfg.project_name)
    wandb.watch(model, log="all")

    mse_list = []
    mae_list = []

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Training loop
    for epoch in range(cfg.training.epochs):
        if epoch % cfg.training.eval_frequency == 0:
            mse, mae = evaluate(model, eval_loader)
            mse_list.append(mse)
            mae_list.append(mae)
            plot_results(mae_list, mse_list, baseline_mae, baseline_mse)
            # Log to Weights & Biases
            wandb.log({"eval/mse": mse, "eval/mae": mae})
            print(f"\n📊 Eval Results — MSE: {mse:.4f}, MAE: {mae:.4f}\n", flush=True)

        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch, y_batch
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        wandb.log({"loss": avg_loss})
        print(f"Epoch {epoch+1}/{cfg.training.epochs} - Loss: {avg_loss:.4f}")

    wandb.finish()
    return model, mae_list, mse_list
