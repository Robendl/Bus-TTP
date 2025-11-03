import torch.multiprocessing as mp

from model.lstm import LSTMFeedforwardCombination

mp.set_start_method("spawn", force=True)

import pandas as pd
import os
import hydra
import torch
import numpy as np
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
from itertools import chain

from data.dataset_bundle import DatasetBundle
from config.config import Config
from data.data_conversions import data_conversions, load_route_lookup
from data.data_processing import create_dataloaders
from model.mlp import MLP
import config.paths as paths
from train.eval import evaluate

os.environ["WANDB_MODE"] = "disabled"
os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    print(f"Using dataset: {cfg.dataset.time}")
    if cfg.pre_data_conversions:
        data_conversions(cfg)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR, cfg)
    output_dir = HydraConfig.get().run.dir
    num_workers = 4 if device.type == 'cuda' else 0

    route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_seq)

    rng = np.random.default_rng(seed=cfg.training.random_state)
    X = dataset_bundle.test.x

    is_route_sequence = True
    lstm_input_dim = next(iter(route_lookup.values())).shape[1]
    ff_input_dim = dataset_bundle.train.x.shape[1] - 3
    model = LSTMFeedforwardCombination(cfg, lstm_input_dim, ff_input_dim)
    model.to(device)

    if cfg.dataset.use_subset:
        path = "outputs/2025-09-23/11-17-03/LSTM.pth"
        # path = "outputs/2025-09-21/20-12-28/MLP.pth"
    else:
        path = "outputs/2025-09-23/13-45-45/LSTM.pth"
        # path = "outputs/2025-09-21/22-26-58/MLP.pth"
    model.load_state_dict(torch.load(path))

    n_repeats = 3

    train_loader, val_loader, test_loader = create_dataloaders(
        cfg, dataset_bundle, route_lookup, is_route_sequence, num_workers
    )
    baseline_metrics, *_ = evaluate(cfg, model, test_loader, device, verbose=False)
    baseline_mae, baseline_mape, baseline_rmse = baseline_metrics

    results = []

    for is_trip, feature in tqdm(reversed(list(
            chain(((True, f) for f in cfg.dataset.time_feature_names),
                  ((False, f) for f in cfg.dataset.route_feature_names),))),
            total=len(cfg.dataset.time_feature_names) + len(cfg.dataset.route_feature_names),
            disable=False
    ):
        print(f"Feature: {feature}")
        deltas_mae, deltas_mape, deltas_rmse = [], [], []
        for rep in range(n_repeats):
            X_perm = X.copy()
            permuted_rl = route_lookup

            if is_trip:
                perm = rng.permutation(len(X))
                X_perm[feature] = X[feature].values[perm]
            else:
                feature_idx = cfg.dataset.route_feature_names.index(feature)

                all_values = np.concatenate([v[:, feature_idx] for v in route_lookup.values()])

                perm = rng.permutation(len(all_values))
                shuffled_values = all_values[perm]

                permuted_rl = {}
                offset = 0
                for key, sequence in route_lookup.items():
                    n = sequence.shape[0]
                    mat_perm = sequence.copy()
                    mat_perm[:, feature_idx] = shuffled_values[offset:offset + n]
                    permuted_rl[key] = mat_perm
                    offset += n

            dataset_bundle.test.x = X_perm
            train_loader, val_loader, test_loader = create_dataloaders(
                cfg, dataset_bundle, permuted_rl, is_route_sequence, num_workers
            )
            (mae, mape, rmse), *_ = evaluate(cfg, model, test_loader, device, verbose=False)

            deltas_mae.append(mae - baseline_mae)
            deltas_mape.append(mape - baseline_mape)
            deltas_rmse.append(rmse - baseline_rmse)

        results.append({
            "features": feature,
            "mean_delta_mae": float(np.mean(deltas_mae)),
            "std_delta_mae": float(np.std(deltas_mae)),
            "mean_delta_mape": float(np.mean(deltas_mape)),
            "std_delta_mape": float(np.std(deltas_mape)),
            "mean_delta_rmse": float(np.mean(deltas_rmse)),
            "std_delta_rmse": float(np.std(deltas_rmse)),
            "baseline_mae": baseline_mae,
            "baseline_mape": baseline_mape,
            "baseline_rmse": baseline_rmse,
        })

        df_results = pd.DataFrame(results)
        df_results.to_csv(output_dir + f"/pfi_results.csv")

    print(pd.DataFrame(results))


if __name__ == "__main__":
    main()
