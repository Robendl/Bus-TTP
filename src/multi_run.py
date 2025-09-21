import torch.multiprocessing as mp
from tqdm import tqdm

from feature_selection.correlation_analysis import correlation_analysis
from plot.analysis import validation_analysis

mp.set_start_method("spawn", force=True)
import pickle
import hydra
import torch
import numpy as np
from hydra.core.hydra_config import HydraConfig

from data.dataset_bundle import DatasetBundle
from data.mapping_dataset import seq_collate_fn, aggr_collate_fn
from plot.plot import plot_tac, scores_boxplot
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

def run_training(cfg, model, route_lookup, dataset_bundle, num_workers, cfg_optim, device, output_dir, is_route_sequence):
    train_loader, val_loader, test_loader = create_dataloaders(cfg, dataset_bundle, route_lookup,
                                                               is_route_sequence, num_workers)
    train_losses, val_losses, val_id_targets, val_mae = train_model(cfg, model, train_loader, val_loader, cfg_optim, device)

    model_dir = f"{output_dir}/{model.name}"
    os.makedirs(model_dir, exist_ok=True)
    val_dir = model_dir + "_val"
    os.makedirs(val_dir, exist_ok=True)

    val_id_targets.to_parquet(f"{val_dir}/{cfg.dataset.time}_id_targets.parquet")
    print(f"{model.name} Val MAE: {val_mae:.3f}")

    (mae, mape, rmse), abs_accuracies, relative_accuracies, test_id_targets, raw_scores = evaluate(cfg, model, test_loader, device)
    print(f"{model.name} Test MAE: {mae:.3f}, MAPE: {mape:.3f}, RMSE: {rmse:.3f} ")

    np.save(f"{model_dir}/{cfg.dataset.time}_train_losses.npy", train_losses)
    np.save(f"{model_dir}/{cfg.dataset.time}_val_losses.npy", val_losses)

    return abs_accuracies, relative_accuracies, test_id_targets

def load_results(cfg: Config, model_name):
    abs_accuracies = np.load(f"{paths.RESULTS_DIR}{model_name}_{cfg.dataset.time}_abs.npy")
    relative_accuracies = np.load(f"{paths.RESULTS_DIR}{model_name}_{cfg.dataset.time}_rel.npy")
    # id_targets = np.load(f"{baseline_dir}/id_targets.npy")
    return abs_accuracies, relative_accuracies

@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    print(f"Using dataset: {cfg.dataset.time}")
    if cfg.pre_data_conversions:
        data_conversions(cfg)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading time data")
    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR
                        + ("_pca" if cfg.dataset.pca else "")
                        + ("_multi" if cfg.dataset.multi_run else ""))

    id_targets_dict = {}
    output_dir = HydraConfig.get().run.dir
    seq_route_lookup = None
    aggr_route_lookup = None
    num_workers = 4 if device.type == 'cuda' else 0
    print(f"num workers: {num_workers}")
    abs_accuracies_dict = {}
    relative_accuracies_dict = {}

    input_dim = len(cfg.dataset.time_feature_names) + len(cfg.dataset.route_feature_names)
    print("Input dim: ", input_dim) # TODO:

    if cfg.compute_baseline or cfg.train_mlp:
        print("Loading aggregated route lookup", flush=True)
        aggr_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_aggr)

    if cfg.train_lstm:
        print("Loading aggregated route lookup", flush=True)
        print("Loading sequence route lookup", flush=True)
        seq_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_seq)

    n_runs = 5
    original_train_x = dataset_bundle.train.x.copy()
    original_train_y = dataset_bundle.train.y.copy()
    dataset_bundle.test.x.drop(["stop_to_stop_id"], axis=1, inplace=True)
    unique_ids = original_train_x['stop_to_stop_id'].unique()

    targets = []
    predictions_list = []

    for run in range(n_runs):
        cfg.training.random_state = cfg.training.random_state + 1
        rng = np.random.default_rng(seed=cfg.training.random_state)
        shuffled_ids = rng.permutation(unique_ids)

        n_total = len(shuffled_ids)
        n_val = int(n_total * 0.2)

        val_ids = set(shuffled_ids[:n_val])
        train_ids = set(shuffled_ids[n_val:])

        train_mask = original_train_x['stop_to_stop_id'].isin(train_ids)
        val_mask = original_train_x['stop_to_stop_id'].isin(val_ids)

        dataset_bundle.train.x = original_train_x[train_mask].drop(["stop_to_stop_id"], axis=1)
        dataset_bundle.train.y = original_train_y[train_mask]
        dataset_bundle.val.x = original_train_x[val_mask].drop(["stop_to_stop_id"], axis=1)
        dataset_bundle.val.y = original_train_y[val_mask]
        dataset_bundle.val.x = dataset_bundle.val.x.reset_index(drop=True)
        dataset_bundle.val.y = dataset_bundle.val.y.reset_index(drop=True)

        print(f"Run {run + 1}: {len(train_ids)} train IDs, {len(val_ids)} val IDs, seed: {cfg.training.random_state}")

        if cfg.train_mlp:
            input_dim = dataset_bundle.train.x.shape[1] - 2 + next(iter(aggr_route_lookup.values())).shape[1]
            model = MLP(cfg, input_dim)
            model.to(device)
            abs_accuracies, relative_accuracies, id_targets = run_training(cfg, model, aggr_route_lookup,
                                                               dataset_bundle, num_workers, cfg.training.optimizer_mlp,
                                                               device, output_dir, is_route_sequence=False)
            targets = id_targets["target"].to_numpy()
            predictions_list.append(id_targets["prediction"].to_numpy())
            abs_accuracies_dict[model.name] = abs_accuracies
            relative_accuracies_dict[model.name] = relative_accuracies
            id_targets_dict["MLP"] = id_targets
        else:
            model_name = "MLP"
            abs_accuracies, relative_accuracies = load_results(cfg, model_name)
            abs_accuracies_dict[model_name] = abs_accuracies
            relative_accuracies_dict[model_name] = relative_accuracies

        if cfg.train_lstm:
            lstm_input_dim = next(iter(seq_route_lookup.values())).shape[1]
            ff_input_dim = dataset_bundle.train.x.shape[1] - 2
            model = LSTMFeedforwardCombination(cfg, lstm_input_dim, ff_input_dim)
            model.to(device)

            abs_accuracies, relative_accuracies, id_targets = run_training(cfg, model, seq_route_lookup,
                                                               dataset_bundle, num_workers, cfg.training.optimizer_lstm,
                                                               device, output_dir, is_route_sequence=True)
            targets = id_targets["target"].to_numpy()
            predictions_list.append(id_targets["prediction"].to_numpy())
            id_targets_dict[model.name] = id_targets
            abs_accuracies_dict[model.name] = abs_accuracies
            relative_accuracies_dict[model.name] = relative_accuracies
        # else:
            # model_name = "LSTM"
            # abs_accuracies, relative_accuracies = load_results(cfg, model_name)
            # abs_accuracies_dict[model_name] = abs_accuracies
            # relative_accuracies_dict[model_name] = relative_accuracies

    Y = np.stack(predictions_list, axis=0)  # (K, n_samples)
    np.save(f"{output_dir}/mr_predictions", Y)
    np.save(f"{output_dir}/mr_targets", targets)
    y_mean = Y.mean(axis=0)

    bias2 = np.mean((y_mean - targets) ** 2)
    variance = np.mean(Y.var(axis=0))
    mse = np.mean((Y - targets) ** 2)
    print(f"Bias^2={bias2:.3f}, Var={variance:.3f}, MSE={mse:.3f}")
    with open(f"{output_dir}/bias_variance.txt", "w") as f:
        f.write(f"Bias^2={bias2:.3f}, Var={variance:.3f}, MSE={mse:.3f} \n")

if __name__ == "__main__":
    main()
