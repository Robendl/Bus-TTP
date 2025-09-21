import torch.multiprocessing as mp
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

    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR
                                        + ("_val" if cfg.dataset.use_validation else "")
                                        + ("_pca" if cfg.dataset.pca else ""),
                                        cfg.dataset.use_validation)
    output_dir = HydraConfig.get().run.dir
    num_workers = 4 if device.type == 'cuda' else 0

    aggr_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_aggr)
    seq_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_seq)

    route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_aggr)

    trip_feature_groups = [['distance'], ['sin_time', 'cos_time'], ['sin_day', 'cos_day'], ['sin_year', 'cos_year'],
                           ['is_public_holiday'], ['is_school_vacation'], ['excess_circuity']]

    route_feature_groups = [['length'], ['max_speed', 'max_speed_alt'], ['num_entrances'], ['on_road_parking_perc_left'],
                            ['on_road_parking_perc_right'], ['schoolzone_perc'], ['num_crossings'], ['avg_width'], ['min_width'],
                            ['max_width'], ['num_narrowing'], ['narrowing_perc'], ['street_perc', 'cityroad_perc',
                                                                                   'regional_perc','residential_perc', 'local_perc', 'unpaved_perc', 'public_transport_perc',
                                                                                   'rest_area_perc', 'highway_perc', 'motorway_perc'],
                            ['pedestrian', 'agricultural', 'bicycle', 'bus', 'car', 'moped', 'motor_scooter', 'motorcycle', 'trailer', 'truck'], ['traffic_signals']]
    is_route_sequence = False
    rng = np.random.default_rng(seed=cfg.training.random_state)
    X = dataset_bundle.test.x

    input_dim = dataset_bundle.train.x.shape[1] - 3 + next(iter(aggr_route_lookup.values())).shape[1]
    model = MLP(cfg, input_dim)
    model.to(device)
    model.load_state_dict(torch.load("outputs/2025-09-21/22-26-58/MLP.pth"))

    n_repeats = 5

    train_loader, val_loader, test_loader = create_dataloaders(
        cfg, dataset_bundle, route_lookup, is_route_sequence, num_workers
    )
    baseline_metrics, *_ = evaluate(cfg, model, test_loader, device, verbose=False)
    baseline_mae, baseline_mape, baseline_rmse = baseline_metrics

    results = []

    for is_trip, cols in tqdm(
            chain(((True, f) for f in trip_feature_groups),
                  ((False, f) for f in route_feature_groups)),
            total=len(trip_feature_groups) + len(route_feature_groups),
            disable=False
    ):
        deltas_mae, deltas_mape, deltas_rmse = [], [], []
        for rep in range(n_repeats):
            X_perm = X.copy()
            permuted_rl = route_lookup

            if is_trip:
                perm = rng.permutation(len(X))
                for col in cols:
                    X_perm[col] = X[col].values[perm]
            else:
                route_matrix = np.vstack(list(route_lookup.values()))
                cols_idx = [cfg.dataset.route_feature_names.index(c) for c in cols]

                perm = rng.permutation(route_matrix.shape[0])
                route_matrix_perm = route_matrix.copy()
                route_matrix_perm[:, cols_idx] = route_matrix[perm, :][:, cols_idx]

                permuted_rl = {}
                for key, values in zip(route_lookup.keys(), route_matrix_perm):
                    permuted_rl[key] = values.reshape(1, -1)

            dataset_bundle.train.x = X_perm
            train_loader, val_loader, test_loader = create_dataloaders(
                cfg, dataset_bundle, permuted_rl, is_route_sequence, num_workers
            )
            (mae, mape, rmse), *_ = evaluate(cfg, model, test_loader, device, verbose=False)

            deltas_mae.append(mae - baseline_mae)
            deltas_mape.append(mape - baseline_mape)
            deltas_rmse.append(rmse - baseline_rmse)

        results.append({
            "features": cols,
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
