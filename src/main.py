import hydra
import torch
from hydra.core.hydra_config import HydraConfig

from config.config import Config
from data.data_processing import load_data, split_data, scale_data, create_dataloader
from model.mlp import MLP
import config.paths as paths
from plot.plot import plot_results
from train.baseline import get_baseline
from train.train import train_model
from train.eval import test, evaluate

import os
os.environ["WANDB_MODE"] = "disabled"

def load_and_eval(cfg: Config):
    model = MLP(cfg.model.input_dim, cfg.model.mlp.hidden_dims, cfg.model.output_dim)



@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    mlp_model = MLP(cfg.model.input_dim, cfg.model.mlp.hidden_dims, cfg.model.output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    mlp_model.to(device)

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

    train_loader = create_dataloader(cfg, X_train_scaled, y_train, device)
    val_loader = create_dataloader(cfg, X_val_scaled, y_val, device)
    test_loader = create_dataloader(cfg, X_test_scaled, y_test, device)
    output_dir = HydraConfig.get().run.dir

    if cfg.train_mlp:
        print("Starting training...", flush=True)
        mlp_model, mae_list, mse_list = train_model(cfg, mlp_model, train_loader, val_loader, val_baseline_mae, val_baseline_mse, val_y_pred_baseline)
        plot_results(mae_list, mse_list, val_baseline_mae, val_baseline_mse)
        mlp_model.load_state_dict(torch.load(f"{output_dir}/weights_{cfg.training.dataset}.pth"))
    else:
        mlp_model.load_state_dict(torch.load(f"model/weights_{cfg.training.dataset}.pth"))

    mse, mae = test(mlp_model, test_loader, y_test)
    print(f"Test | mse: {mse:.3f}, mae: {mae:.3f} ")
    print(f"Baseline | mse: {test_baseline_mse:.3f}, mae: {test_baseline_mae:.3f}")
    with open(f"{output_dir}/results.txt", "w") as f:
        f.write(f"Test | mse: {mse:.3f}, mae: {mae:.3f}\n")
        f.write(f"Baseline | mse: {test_baseline_mse:.3f}, mae: {test_baseline_mae:.3f}\n")


if __name__ == "__main__":
    main()
