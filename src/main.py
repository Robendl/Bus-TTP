"""Main experiment entry point.

Trains and/or evaluates the four model families (Linear Regression, XGBoost,
MLP, LSTM) on the bus travel-time dataset and writes per-model artifacts,
bootstrapped score intervals, and tolerance-accuracy curves to the Hydra
run directory.
"""
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig

import config.paths as paths
from config.config import Config
from data.build_dataset import build_dataset, load_route_lookup
from data.data_processing import create_dataloaders
from data.dataset_bundle import DatasetBundle
from model.lstm import LSTMFeedforwardCombination
from model.mlp import MLP
from plot.analysis import (
    bootstrap_ci,
    get_od_results,
    paired_significance_test,
    residual_plots,
    validation_analysis,
)
from plot.plot import bootstrap_tac_per_model
from runtime import setup_environment
from train.eval import evaluate
from train.linear_regression import linear_regression
from train.train import train_model
from train.xgboost import train_xgb

setup_environment()


def _train_neural_model(
    cfg: Config,
    model: torch.nn.Module,
    route_lookup,
    dataset_bundle: DatasetBundle,
    optimizer_cfg,
    is_route_sequence: bool,
    device: torch.device,
    output_dir: str,
    num_workers: int,
):
    """Train a neural model, evaluate on the test set, and persist artifacts."""
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg, dataset_bundle, route_lookup, is_route_sequence, num_workers
    )
    train_losses, val_losses, val_id_targets, val_mae, _, _ = train_model(
        cfg, model, train_loader, val_loader, optimizer_cfg, device
    )

    model_dir = Path(output_dir) / model.name
    model_dir.mkdir(parents=True, exist_ok=True)

    (mae, mape, rmse), abs_acc, rel_acc, test_id_targets, *_ = evaluate(
        cfg, model, test_loader, device
    )
    print(f"{model.name} test MAE: {mae:.3f}, MAPE: {mape:.3f}, RMSE: {rmse:.3f}")

    results = test_id_targets.merge(
        dataset_bundle.test.x[["id", "stop_to_stop_id"]], on="id", how="left"
    )
    results.to_parquet(model_dir / "id_targets.parquet")

    od_results = get_od_results(results)
    _, ci_string = bootstrap_ci(od_results, seed=cfg.training.random_state, model_name=model.name)
    print(ci_string)

    if not cfg.dataset.pca:
        residual_plots(cfg, test_id_targets, model_dir, split="test", use_subset=cfg.dataset.use_subset)
    validation_analysis(test_id_targets, model_dir, split="test", use_subset=cfg.dataset.use_subset)

    np.save(model_dir / f"{cfg.dataset.time}_abs.npy", abs_acc)
    np.save(model_dir / f"{cfg.dataset.time}_rel.npy", rel_acc)
    np.save(model_dir / f"{cfg.dataset.time}_train_losses.npy", train_losses)

    if cfg.dataset.use_validation:
        val_dir = Path(f"{model_dir}_val")
        val_dir.mkdir(parents=True, exist_ok=True)
        val_id_targets.to_parquet(val_dir / f"{cfg.dataset.time}_id_targets.parquet")
        validation_analysis(val_id_targets, val_dir, split="val", use_subset=cfg.dataset.use_subset)
        np.save(model_dir / f"{cfg.dataset.time}_val_losses.npy", val_losses)
        print(f"{model.name} val MAE: {val_mae:.3f}")

    return results, od_results["MAE"], ci_string


def _run_linear_regression(cfg: Config, dataset_bundle: DatasetBundle, route_lookup):
    val_mae, (mae, mape, rmse), *_, id_targets = linear_regression(cfg, dataset_bundle, route_lookup)
    results = id_targets.merge(
        dataset_bundle.test.x[["id", "stop_to_stop_id"]], on="id", how="left"
    )
    od_results = get_od_results(results)
    _, ci_string = bootstrap_ci(od_results, seed=cfg.training.random_state, model_name="Linear Regression")
    print(ci_string)
    print(f"Linear Regression test MAE: {mae:.3f}, MAPE: {mape:.3f}, RMSE: {rmse:.3f}")
    return results, ci_string


def _run_xgboost(cfg: Config, dataset_bundle: DatasetBundle, output_dir: str):
    route_df = pd.read_parquet(paths.DATASETS_DIR + cfg.dataset.route_aggr + "_val.parquet")
    xgb_dir = Path(output_dir) / "xgboost"
    xgb_dir.mkdir(parents=True, exist_ok=True)

    id_targets = train_xgb(cfg, dataset_bundle, route_df, xgb_dir)
    results = id_targets.merge(
        dataset_bundle.test.x[["id", "stop_to_stop_id"]], on="id", how="left"
    )
    od_results = get_od_results(results)
    _, ci_string = bootstrap_ci(od_results, seed=cfg.training.random_state, model_name="XGBoost")
    print(ci_string)
    return results, ci_string


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    if cfg.build_dataset:
        build_dataset(cfg)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = HydraConfig.get().run.dir
    num_workers = 4 if device.type == "cuda" else 0
    print(f"Device: {device} | num workers: {num_workers}")

    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR, cfg)

    aggr_route_lookup = None
    if cfg.compute_baseline or cfg.train_mlp or cfg.fit_xgboost:
        aggr_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_aggr)

    results_dict: Dict[str, pd.DataFrame] = {}
    ci_strings = []
    od_mae_per_model: Dict[str, pd.Series] = {}

    if cfg.compute_baseline and not cfg.dataset.pca:
        results_dict["Linear Regression"], ci = _run_linear_regression(cfg, dataset_bundle, aggr_route_lookup)
        ci_strings.append(ci)

    if cfg.fit_xgboost:
        results_dict["XGBoost"], ci = _run_xgboost(cfg, dataset_bundle, output_dir)
        ci_strings.append(ci)

    if cfg.train_mlp:
        input_dim = dataset_bundle.train.x.shape[1] - 3 + next(iter(aggr_route_lookup.values())).shape[1]
        model = MLP(cfg, input_dim).to(device)
        results, od_mae, ci = _train_neural_model(
            cfg, model, aggr_route_lookup, dataset_bundle, cfg.training.optimizer_mlp,
            is_route_sequence=False, device=device, output_dir=output_dir, num_workers=num_workers,
        )
        results_dict[model.name] = results
        od_mae_per_model[model.name] = od_mae
        ci_strings.append(ci)

    if cfg.train_lstm:
        seq_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_seq)
        lstm_input_dim = next(iter(seq_route_lookup.values())).shape[1]
        ff_input_dim = dataset_bundle.train.x.shape[1] - 3
        model = LSTMFeedforwardCombination(cfg, lstm_input_dim, ff_input_dim).to(device)
        results, od_mae, ci = _train_neural_model(
            cfg, model, seq_route_lookup, dataset_bundle, cfg.training.optimizer_lstm,
            is_route_sequence=True, device=device, output_dir=output_dir, num_workers=num_workers,
        )
        results_dict[model.name] = results
        od_mae_per_model[model.name] = od_mae
        ci_strings.append(ci)

    with open(Path(output_dir) / "final_scores.txt", "w") as f:
        f.write("\n".join(ci_strings) + "\n")

    if "MLP" in od_mae_per_model and "LSTM" in od_mae_per_model:
        ttest_p, wilcoxon_p = paired_significance_test(
            od_mae_per_model["MLP"], od_mae_per_model["LSTM"]
        )
        with open(Path(output_dir) / "paired_significance.txt", "w") as f:
            f.write(f"Paired t-test p = {ttest_p:.5f}\n")
            f.write(f"Wilcoxon p = {wilcoxon_p:.5f}\n")
        print(f"Paired t-test p = {ttest_p:.5f} | Wilcoxon p = {wilcoxon_p:.5f}")

    if results_dict:
        absolute_margins = np.arange(0, cfg.plot.margins_max, cfg.plot.step_size)
        relative_margins = np.arange(0, cfg.plot.percentages_max, cfg.plot.step_size)
        bootstrap_tac_per_model(results_dict, absolute_margins, cfg.training.random_state, output_dir, percentage=False)
        bootstrap_tac_per_model(results_dict, relative_margins, cfg.training.random_state, output_dir, percentage=True)


if __name__ == "__main__":
    main()
