import hydra
import torch
from hydra.core.hydra_config import HydraConfig

from config.config import Config
from data.data import load_data, split_data, scale_data, create_dataloader
from model.mlp import MLP
import config.paths as paths
from plot.plot import plot_results
from train.baseline import get_baseline
from train.train import train_model
from train.eval import test

import os
os.environ["WANDB_MODE"] = "disabled"

@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    print(f"Loading data... ({cfg.training.dataset})", flush=True)
    df = load_data(paths.DATASETS_DIR + cfg.training.dataset)
    df.drop(columns=["recordeddeparturetime"], inplace=True)
    print("Filling 0's")
    df.fillna(0, inplace=True)
    print("Splitting data")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, cfg.training.val_size, cfg.training.test_size, cfg.training.random_state)
    X_train_scaled, X_val_scaled, X_test_scaled = scale_data(X_train, X_val, X_test)

    # correlation_analysis(X_train, y_train)
    # plot_distribution(X_train, y_train)

    X_train_scaled.drop(columns=["stop_to_stop_id"], inplace=True)
    X_val_scaled.drop(columns=["stop_to_stop_id"], inplace=True)
    X_test_scaled.drop(columns=["stop_to_stop_id"], inplace=True)

    print("Computing baseline", flush=True)
    val_baseline_mae, val_baseline_mse, val_y_pred_baseline, test_baseline_mae, test_baseline_mse, test_y_pred_baseline = get_baseline(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
    print(f"Baseline: MAE: {val_baseline_mae:.2f} MSE: {val_baseline_mse:.2f}")

    model = MLP(X_train_scaled.shape[1], cfg.model.mlp.hidden_dims, cfg.model.output_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)

    train_loader = create_dataloader(cfg, X_train_scaled, y_train, device)
    val_loader = create_dataloader(cfg, X_val_scaled, y_val, device)
    test_loader = create_dataloader(cfg, X_test_scaled, y_test, device)

    print("Starting training...", flush=True)
    model, mae_list, mse_list = train_model(cfg, model, train_loader, val_loader, val_baseline_mae, val_baseline_mse, val_y_pred_baseline)

    plot_results(mae_list, mse_list, val_baseline_mae, val_baseline_mse)

    output_dir = HydraConfig.get().run.dir
    model.load_state_dict(torch.load(f"{output_dir}/weights_{cfg.training.dataset}.pth"))
    mse, mae = test(model, test_loader, y_test)
    print(f"Test | mse: {mse:.3f}, mae: {mae:.3f} ")
    print(f"Baseline | mse: {test_baseline_mse:.3f}, mae: {test_baseline_mae:.3f}")

if __name__ == "__main__":
    main()
