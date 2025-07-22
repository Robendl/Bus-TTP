import torch.multiprocessing as mp
from tqdm import tqdm

from feature_selection.correlation_analysis import correlation_analysis

mp.set_start_method("spawn", force=True)
import pickle
import hydra
import torch
import numpy as np
from hydra.core.hydra_config import HydraConfig

from data.dataset_bundle import DatasetBundle
from data.mapping_dataset import seq_collate_fn, aggr_collate_fn
from plot.plot import plot_tac
from config.config import Config
from data.data_conversions import data_conversions, load_route_lookup
from data.data_processing import create_dataloaders
from model.lstm import LSTMFeedforwardCombination
from model.mlp import MLP
import config.paths as paths
from train.linear_regression import linear_regression
from train.train import train_model
from train.eval import evaluate

import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["HYDRA_FULL_ERROR"] = "1"

def run_training(cfg, model, route_lookup, collate_fn, dataset_bundle, num_workers, learning_rate, device, output_dir):
    train_loader, val_loader, test_loader = create_dataloaders(cfg, dataset_bundle, route_lookup,
                                                               collate_fn, num_workers)
    train_losses, val_losses, best_id_targets = train_model(cfg, model, train_loader, val_loader, learning_rate, device)
    best_id_targets.to_parquet(f"{output_dir}/{model.name}_{cfg.dataset.time}_id_targets.parquet")
    np.save(f"{output_dir}/{model.name}_{cfg.dataset.time}_id_targets.npy", best_id_targets)

    model.load_state_dict(torch.load(f"{output_dir}/{model.name}.pth"))
    mae, abs_accuracies, relative_accuracies, _ = evaluate(cfg, model, test_loader, device)
    print(f"{model.name} Test MAE: {mae:.3f} ")

    mae_path = os.path.join(output_dir, f"{model.name}_mae.txt")
    with open(mae_path, "w") as f:
        f.write(f"Test MAE: {mae:.3f}\n")

    if cfg.save_results:
        accuracies_dir = paths.RESULTS_DIR
        np.save(f"{accuracies_dir}/{model.name}_{cfg.dataset.time}_abs.npy", abs_accuracies)
        np.save(f"{accuracies_dir}/{model.name}_{cfg.dataset.time}_rel.npy", relative_accuracies)

    np.save(f"{output_dir}/{model.name}_{cfg.dataset.time}_abs.npy", abs_accuracies)
    np.save(f"{output_dir}/{model.name}_{cfg.dataset.time}_rel.npy", relative_accuracies)
    np.save(f"{output_dir}/{model.name}_{cfg.dataset.time}_train_losses.npy", train_losses)
    np.save(f"{output_dir}/{model.name}_{cfg.dataset.time}_val_losses.npy", val_losses)

    return abs_accuracies, relative_accuracies

def load_results(cfg: Config, model_name):
    abs_accuracies = np.load(f"{paths.RESULTS_DIR}{model_name}_{cfg.dataset.time}_abs.npy")
    relative_accuracies = np.load(f"{paths.RESULTS_DIR}{model_name}_{cfg.dataset.time}_rel.npy")
    return abs_accuracies, relative_accuracies

@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    print(f"Using dataset: {cfg.dataset.time}")
    if cfg.pre_data_conversions:
        data_conversions(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading time data")
    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR)
    # dataset_bundle.train.x = dataset_bundle.train.x.iloc[:8000]
    # dataset_bundle.train.y = dataset_bundle.train.y.iloc[:8000]
    # dataset_bundle.val.x = dataset_bundle.val.x.iloc[:8000]
    # dataset_bundle.val.y = dataset_bundle.val.y.iloc[:8000]
    # dataset_bundle.test.x = dataset_bundle.test.x.iloc[:8000]
    # dataset_bundle.test.y = dataset_bundle.test.y.iloc[:8000]

    # correlation_analysis(X_train, y_train)
    # plot_distribution(X_train, y_train)

    output_dir = HydraConfig.get().run.dir
    seq_route_lookup = None
    aggr_route_lookup = None
    num_workers = 4 if device.type == 'cuda' else 0
    print(f"num workers: {num_workers}")
    abs_accuracies_dict = {}
    relative_accuracies_dict = {}

    if cfg.compute_baseline or cfg.train_mlp:
        print("Loading aggregated route lookup", flush=True)
        aggr_route_lookup = load_route_lookup(paths.DATASETS_DIR + cfg.dataset.route_aggr)

    if cfg.compute_baseline:
        print("Computing baseline", flush=True)
        lr_val_mae, lr_test_mae, abs_accuracies, relative_accuracies = linear_regression(cfg, dataset_bundle, aggr_route_lookup)
        print(f"Baseline MAE | val: {lr_val_mae:.2f}, test: {lr_test_mae:.2f}", flush=True)
        abs_accuracies_dict["Linear regression"] = abs_accuracies
        relative_accuracies_dict["Linear regression"] = relative_accuracies

    if cfg.train_mlp:
        model = MLP(cfg.model.input_dim, cfg.model.mlp.hidden_dims, cfg.model.output_dim)
        model.to(device)
        abs_accuracies, relative_accuracies = run_training(cfg, model, aggr_route_lookup, aggr_collate_fn,
                                                           dataset_bundle, num_workers, cfg.model.mlp.learning_rate,
                                                           device, output_dir)
        abs_accuracies_dict[model.name] = abs_accuracies
        relative_accuracies_dict[model.name] = relative_accuracies
    else:
        model_name = "MLP"
        abs_accuracies, relative_accuracies = load_results(cfg, model_name)
        abs_accuracies_dict[model_name] = abs_accuracies
        relative_accuracies_dict[model_name] = relative_accuracies

    if cfg.train_lstm:
        model = LSTMFeedforwardCombination(len(cfg.dataset.route_feature_names), cfg.model.lstm.hidden_dim, len(cfg.dataset.time_feature_names), cfg.model.lstm.ff_hidden_dim)
        model.to(device)

        print("Loading sequence route lookup", flush=True)
        seq_route_lookup = load_route_lookup(paths.DATASETS_DIR + cfg.dataset.route_seq)

        abs_accuracies, relative_accuracies = run_training(cfg, model, seq_route_lookup, seq_collate_fn,
                                                           dataset_bundle, num_workers, cfg.model.lstm.learning_rate,
                                                           device, output_dir)
        abs_accuracies_dict[model.name] = abs_accuracies
        relative_accuracies_dict[model.name] = relative_accuracies
    else:
        model_name = "LSTM"
        abs_accuracies, relative_accuracies = load_results(cfg, model_name)
        abs_accuracies_dict[model_name] = abs_accuracies
        relative_accuracies_dict[model_name] = relative_accuracies

    margins = np.arange(1, cfg.plot.margins_max, cfg.plot.step_size)
    plot_tac(margins, abs_accuracies_dict, 's', output_dir)
    margins = np.arange(1, cfg.plot.percentages_max, cfg.plot.step_size)
    plot_tac(margins, relative_accuracies_dict, 'p', output_dir)


if __name__ == "__main__":
    main()
