"""Bias / variance analysis through repeated train-validation resplits.

Reshuffles the train/val split N times (by OD pair) and refits the selected
model on each split. The per-sample predictions across runs are aggregated
into bias^2 and variance estimates.
"""
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig

import config.paths as paths
from config.config import Config
from data.build_dataset import build_dataset, load_route_lookup
from data.data_processing import create_dataloaders
from data.dataset_bundle import DatasetBundle
from model.lstm import LSTMFeedforwardCombination
from model.mlp import MLP
from runtime import setup_environment
from train.eval import evaluate
from train.train import train_model

setup_environment()

VAL_FRACTION = 0.2


def _resplit_train_val(dataset_bundle: DatasetBundle, original_x, original_y, seed: int):
    """Replace train/val splits in-place with a fresh OD-disjoint resplit."""
    rng = np.random.default_rng(seed=seed)
    unique_ids = original_x["stop_to_stop_id"].unique()
    shuffled_ids = rng.permutation(unique_ids)
    n_val = int(len(shuffled_ids) * VAL_FRACTION)
    val_ids = set(shuffled_ids[:n_val])
    train_ids = set(shuffled_ids[n_val:])

    train_mask = original_x["stop_to_stop_id"].isin(train_ids)
    val_mask = original_x["stop_to_stop_id"].isin(val_ids)

    dataset_bundle.train.x = original_x[train_mask].drop(["stop_to_stop_id"], axis=1)
    dataset_bundle.train.y = original_y[train_mask]
    dataset_bundle.val.x = original_x[val_mask].drop(["stop_to_stop_id"], axis=1).reset_index(drop=True)
    dataset_bundle.val.y = original_y[val_mask].reset_index(drop=True)
    return len(train_ids), len(val_ids)


def _train_one_run(
    cfg: Config, model, route_lookup, dataset_bundle, optimizer_cfg,
    device, output_dir, is_route_sequence, num_workers,
):
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg, dataset_bundle, route_lookup, is_route_sequence, num_workers
    )
    train_losses, val_losses, val_id_targets, val_mae, _, _ = train_model(
        cfg, model, train_loader, val_loader, optimizer_cfg, device
    )
    print(f"{model.name} val MAE: {val_mae:.3f}")

    model_dir = Path(output_dir) / model.name
    model_dir.mkdir(parents=True, exist_ok=True)
    val_dir = Path(f"{model_dir}_val")
    val_dir.mkdir(parents=True, exist_ok=True)

    val_id_targets.to_parquet(val_dir / f"{cfg.dataset.time}_id_targets.parquet")
    np.save(model_dir / f"{cfg.dataset.time}_train_losses.npy", train_losses)
    np.save(model_dir / f"{cfg.dataset.time}_val_losses.npy", val_losses)

    (mae, mape, rmse), _, _, test_id_targets, _, _ = evaluate(cfg, model, test_loader, device)
    print(f"{model.name} test MAE: {mae:.3f}, MAPE: {mape:.3f}, RMSE: {rmse:.3f}")
    return test_id_targets


def _build_model(cfg: Config, dataset_bundle, route_lookup, is_route_sequence: bool):
    route_dim = next(iter(route_lookup.values())).shape[1]
    if is_route_sequence:
        ff_input_dim = dataset_bundle.train.x.shape[1] - 2
        return LSTMFeedforwardCombination(cfg, route_dim, ff_input_dim)
    input_dim = dataset_bundle.train.x.shape[1] - 2 + route_dim
    return MLP(cfg, input_dim)


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    if cfg.build_dataset:
        build_dataset(cfg)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = HydraConfig.get().run.dir
    num_workers = 4 if device.type == "cuda" else 0
    print(f"Device: {device} | num workers: {num_workers}")

    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR)

    if cfg.train_lstm:
        is_route_sequence = True
        route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_seq)
        optimizer_cfg = cfg.training.optimizer_lstm
    else:
        is_route_sequence = False
        route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_aggr)
        optimizer_cfg = cfg.training.optimizer_mlp

    original_train_x = dataset_bundle.train.x.copy()
    original_train_y = dataset_bundle.train.y.copy()
    dataset_bundle.test.x.drop(["stop_to_stop_id"], axis=1, inplace=True)

    n_runs = 5
    targets, predictions = None, []
    for run in range(n_runs):
        seed = cfg.training.random_state + run + 1
        n_train, n_val = _resplit_train_val(dataset_bundle, original_train_x, original_train_y, seed)
        print(f"Run {run + 1}/{n_runs}: {n_train} train OD pairs, {n_val} val OD pairs (seed={seed})")

        model = _build_model(cfg, dataset_bundle, route_lookup, is_route_sequence).to(device)
        test_id_targets = _train_one_run(
            cfg, model, route_lookup, dataset_bundle, optimizer_cfg,
            device, output_dir, is_route_sequence, num_workers,
        )
        targets = test_id_targets["target"].to_numpy()
        predictions.append(test_id_targets["prediction"].to_numpy())

    predictions = np.stack(predictions, axis=0)
    np.save(Path(output_dir) / "mr_predictions.npy", predictions)
    np.save(Path(output_dir) / "mr_targets.npy", targets)

    mean_pred = predictions.mean(axis=0)
    bias2 = np.mean((mean_pred - targets) ** 2)
    variance = np.mean(predictions.var(axis=0))
    mse = np.mean((predictions - targets) ** 2)
    summary = f"Bias^2={bias2:.3f}, Var={variance:.3f}, MSE={mse:.3f}"
    print(summary)
    (Path(output_dir) / "bias_variance.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
