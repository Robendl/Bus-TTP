from itertools import chain

import pandas as pd
import torch.multiprocessing as mp
from tqdm import tqdm

from feature_selection.correlation_analysis import correlation_analysis
from plot.analysis import validation_analysis, get_od_results, bootstrap_ci, paired_significance_test
from train.xgboost import xgboost_gridsearch, train_xgb

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

    (mae, mape, rmse), abs_accuracies, relative_accuracies, test_id_targets, raw_scores = evaluate(cfg, model, test_loader, device)
    test_id_targets.to_parquet(f"{model_dir}/{cfg.dataset.time}_id_targets.parquet")

    results = test_id_targets.merge(dataset_bundle.test.x[["id", "stop_to_stop_id"]], on="id", how="left")
    od_results = get_od_results(results)
    bootstrap, result_string = bootstrap_ci(od_results, seed=cfg.training.random_state, model_name=model.name)
    print(result_string)
    print(f"{model.name} Test MAE: {mae:.3f}, MAPE: {mape:.3f}, RMSE: {rmse:.3f} ")
    validation_analysis(test_id_targets, model_dir, split="test", use_subset=cfg.dataset.use_subset)

    mae_path = os.path.join(output_dir, f"{model.name}_mae.txt")
    with open(mae_path, "w") as f:
        if cfg.dataset.use_validation:
            f.write(f"Val MAE: {val_mae:.3f}\n")
        f.write(f"Test MAE: {mae:.3f}, MAPE: {mape:.3f}, RMSE; {rmse:.3f} \n")

    if cfg.save_results:
        accuracies_dir = paths.RESULTS_DIR
        np.save(f"{accuracies_dir}/{model.name}_{cfg.dataset.time}_abs.npy", abs_accuracies)
        np.save(f"{accuracies_dir}/{model.name}_{cfg.dataset.time}_rel.npy", relative_accuracies)

    np.save(f"{model_dir}/{cfg.dataset.time}_abs.npy", abs_accuracies)
    np.save(f"{model_dir}/{cfg.dataset.time}_rel.npy", relative_accuracies)
    np.save(f"{model_dir}/{cfg.dataset.time}_train_losses.npy", train_losses)

    if cfg.dataset.use_validation:
        val_dir = model_dir + "_val"
        os.makedirs(val_dir, exist_ok=True)
        val_id_targets.to_parquet(f"{val_dir}/{cfg.dataset.time}_id_targets.parquet")
        print(f"{model.name} Val MAE: {val_mae:.3f}")
        validation_analysis(val_id_targets, val_dir, split="val", use_subset=cfg.dataset.use_subset)
        np.save(f"{model_dir}/{cfg.dataset.time}_val_losses.npy", val_losses)

    return abs_accuracies, relative_accuracies, raw_scores, od_results["MAE"], result_string

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
                                        + ("_val" if cfg.dataset.use_validation else "")
                                        + ("_pca" if cfg.dataset.pca else ""),
                                        cfg.dataset.use_validation)
    print(dataset_bundle.train.x.shape)

    output_dir = HydraConfig.get().run.dir
    num_workers = 4 if device.type == 'cuda' else 0
    print(f"num workers: {num_workers}")

    aggr_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_aggr)
    seq_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_seq)

    path = paths.DATASETS_DIR + cfg.dataset.route_aggr
    aggr_route_df = pd.read_parquet(path + ("_val" if cfg.dataset.use_validation else "") + ".parquet")

    aggregated = True
    trip_feature_groups = [['distance'], ['sin_time', 'cos_time'], ['sin_day', 'cos_day'], ['sin_year', 'cos_year'],
                           ['is_public_holiday'], ['is_school_vacation'], ['excess_circuity']]

    route_feature_groups = [['length'], ['max_speed', 'max_speed_alt'], ['num_entrances'], ['on_road_parking_perc_left'],
                            ['on_road_parking_perc_right'], ['schoolzone_perc'], ['num_crossings'], ['avg_width'], ['min_width'],
                            ['max_width'], ['num_narrowing'], ['narrowing_perc'], ['street_perc', 'cityroad_perc',
                                                                                   'regional_perc','residential_perc', 'local_perc', 'unpaved_perc', 'public_transport_perc',
                                                                                   'rest_area_perc', 'highway_perc', 'motorway_perc'],
                            ['pedestrian', 'agricultural', 'bicycle', 'bus', 'car', 'moped', 'motor_scooter', 'motorcycle', 'trailer', 'truck'], ['traffic_signals']]

    np.random.default_rng(seed=cfg.training.random_state)
    X = dataset_bundle.test.x
    y = dataset_bundle.test.y
    for is_trip, cols in chain(
            ((True, f) for f in trip_feature_groups),
                ((False, f) for f in route_feature_groups)):

        X_perm = X.copy()
        route_perm = aggr_route_df.copy()

        if is_trip:
            perm = np.random.permutation(len(X))
            for col in cols:
                X_perm[col] = X[col].values[perm]
        else:
            perm = np.random.permutation(len(aggr_route_df))

        route_lookup = {}
        for hash_val, group in route_perm.groupby("route_seq_hash"):
            group.drop(columns=["route_seq_hash"], inplace=True)
            values = group.values.astype(np.float32)
            if aggregated:
                values = values.reshape(1, -1)
            route_lookup[str(hash_val)] = values

        input_dim = dataset_bundle.train.x.shape[1] - 3 + next(iter(aggr_route_lookup.values())).shape[1]
        model = MLP(cfg, input_dim)
        model.to(device)
        model.load_state_dict(torch.load(f"outputs/2025-09-21/20-12-28/MLP.pth"))

        is_route_sequence = not aggregated
        train_loader, val_loader, test_loader = create_dataloaders(cfg, dataset_bundle, route_lookup,
                                                                   is_route_sequence, num_workers)
        (mae, mape, rmse), _, _, _, _ = evaluate(cfg, model, test_loader, device)






if __name__ == "__main__":
    main()
