"""Permutation Feature Importance (PFI) for the MLP and LSTM models.

Permutes one feature group at a time and measures the resulting drop in
test-set MAE / MAPE / RMSE. Trip features are permuted across rows; route
features are permuted either across aggregated routes (MLP) or across
all route segments (LSTM).
"""
from pathlib import Path
from typing import Dict, List, Sequence

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

import config.paths as paths
from config.config import Config
from data.data_conversions import data_conversions, load_route_lookup
from data.data_processing import create_dataloaders
from data.dataset_bundle import DatasetBundle
from model.lstm import LSTMFeedforwardCombination
from model.mlp import MLP
from runtime import setup_environment
from train.eval import evaluate

setup_environment()


def _permute_trip_features(
    df: pd.DataFrame,
    feature_group: Sequence[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    perm = rng.permutation(len(df))
    df_perm = df.copy()
    for col in feature_group:
        df_perm[col] = df[col].values[perm]
    return df_perm


def _permute_route_lookup_aggr(
    route_lookup: Dict[str, np.ndarray],
    feature_indices: List[int],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Permute features in an aggregated route lookup (one row per route)."""
    keys = list(route_lookup.keys())
    matrix = np.vstack([route_lookup[k] for k in keys])
    perm = rng.permutation(matrix.shape[0])
    matrix_perm = matrix.copy()
    matrix_perm[:, feature_indices] = matrix[perm][:, feature_indices]
    return {k: row.reshape(1, -1) for k, row in zip(keys, matrix_perm)}


def _permute_route_lookup_seq(
    route_lookup: Dict[str, np.ndarray],
    feature_indices: List[int],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Permute features across every segment of every route sequence."""
    pooled = np.concatenate([v[:, feature_indices] for v in route_lookup.values()])
    pooled_perm = pooled[rng.permutation(len(pooled))]

    permuted = {}
    offset = 0
    for key, sequence in route_lookup.items():
        n = sequence.shape[0]
        new_seq = sequence.copy()
        new_seq[:, feature_indices] = pooled_perm[offset:offset + n]
        permuted[key] = new_seq
        offset += n
    return permuted


def _build_model(cfg: Config, dataset_bundle: DatasetBundle, route_lookup, is_route_sequence: bool):
    route_dim = next(iter(route_lookup.values())).shape[1]
    if is_route_sequence:
        ff_input_dim = dataset_bundle.train.x.shape[1] - 3
        return LSTMFeedforwardCombination(cfg, route_dim, ff_input_dim)
    input_dim = dataset_bundle.train.x.shape[1] - 3 + route_dim
    return MLP(cfg, input_dim)


def _trip_feature_groups() -> List[List[str]]:
    return [
        ["distance"],
        ["sin_time", "cos_time"],
        ["sin_day", "cos_day"],
        ["sin_year", "cos_year"],
        ["is_public_holiday"],
        ["is_school_vacation"],
        ["excess_circuity"],
    ]


def compute_pfi(
    cfg: Config,
    model: torch.nn.Module,
    dataset_bundle: DatasetBundle,
    route_lookup: Dict[str, np.ndarray],
    device: torch.device,
    is_route_sequence: bool,
    output_dir: str,
    n_repeats: int = 3,
    num_workers: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.training.random_state)
    X_test = dataset_bundle.test.x.copy()

    *_, test_loader = create_dataloaders(
        cfg, dataset_bundle, route_lookup, is_route_sequence, num_workers
    )
    (baseline_mae, baseline_mape, baseline_rmse), *_ = evaluate(
        cfg, model, test_loader, device, verbose=False
    )

    permute_route = (
        _permute_route_lookup_seq if is_route_sequence else _permute_route_lookup_aggr
    )
    trip_groups = _trip_feature_groups()
    route_groups = [[f] for f in cfg.dataset.route_feature_names]
    feature_groups = [("trip", g) for g in trip_groups] + [("route", g) for g in route_groups]

    results = []
    for kind, group in tqdm(feature_groups, desc="PFI"):
        deltas_mae, deltas_mape, deltas_rmse = [], [], []
        for _ in range(n_repeats):
            if kind == "trip":
                dataset_bundle.test.x = _permute_trip_features(X_test, group, rng)
                permuted_lookup = route_lookup
            else:
                dataset_bundle.test.x = X_test
                feature_idx = [cfg.dataset.route_feature_names.index(f) for f in group]
                permuted_lookup = permute_route(route_lookup, feature_idx, rng)

            *_, test_loader = create_dataloaders(
                cfg, dataset_bundle, permuted_lookup, is_route_sequence, num_workers
            )
            (mae, mape, rmse), *_ = evaluate(cfg, model, test_loader, device, verbose=False)
            deltas_mae.append(mae - baseline_mae)
            deltas_mape.append(mape - baseline_mape)
            deltas_rmse.append(rmse - baseline_rmse)

        results.append({
            "features": group,
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
        pd.DataFrame(results).to_csv(Path(output_dir) / "pfi_results.csv", index=False)

    dataset_bundle.test.x = X_test
    return pd.DataFrame(results)


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    if cfg.pre_data_conversions:
        data_conversions(cfg)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = HydraConfig.get().run.dir
    num_workers = 4 if device.type == "cuda" else 0

    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR, cfg)

    if cfg.train_lstm:
        is_route_sequence = True
        route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_seq)
    else:
        is_route_sequence = False
        route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_aggr)

    model = _build_model(cfg, dataset_bundle, route_lookup, is_route_sequence).to(device)
    checkpoint_path = cfg.eval.checkpoint_path
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    pfi_df = compute_pfi(
        cfg, model, dataset_bundle, route_lookup, device, is_route_sequence, output_dir,
        num_workers=num_workers,
    )
    print(pfi_df)


if __name__ == "__main__":
    main()
