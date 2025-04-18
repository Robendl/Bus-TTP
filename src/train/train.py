import torch
import torch.nn as nn
import wandb
from omegaconf import omegaconf
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple

from config.config import Config
from train.eval import evaluate


def train_model(cfg: Config, model, train_loader, eval_loader, X_val=None, y_val=None):
    # Initialize Weights & Biases
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project=cfg.project_name)
    wandb.watch(model, log="all")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Training loop
    for epoch in range(cfg.training.epochs):
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
        wandb.log({"epoch": epoch, "loss": avg_loss})
        print(f"Epoch {epoch+1}/{cfg.training.epochs} - Loss: {avg_loss:.4f}")

        if epoch % (cfg.training.eval_frequency * cfg.training.epochs) == 0:
            evaluate(model, eval_loader)

    wandb.finish()
    return model
