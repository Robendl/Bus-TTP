import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import hydra
import numpy as np
import os

from config.config import Config
import config.paths as paths
from data.data_conversions import load_route_lookup
from data.data_processing import create_dataloaders
from data.dataset_bundle import DatasetBundle
from model.lstm import LSTMFeedforwardCombination
from train.eval import evaluate

mp.set_start_method("spawn", force=True)
os.environ["WANDB_MODE"] = "disabled"
os.environ["HYDRA_FULL_ERROR"] = "1"


def bootstrap_od_errors_per_route(results: pd.DataFrame, n_boot=100, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    od_groups = results.groupby("stop_to_stop_id")
    od_stats = []

    for od_id, df in tqdm(od_groups, desc="Bootstrapping OD errors"):
        errors = ((df["prediction"].to_numpy() - df["target"].to_numpy()) /
                  df["target"].replace(0, np.nan).to_numpy()) * 100
        errors = errors[~np.isnan(errors)]
        n = len(errors)

        mean_target = df["target"].mean()
        mean_prediction = df["prediction"].mean()

        if n < 2:
            mean_error = np.nanmean(errors)
            od_stats.append({
                "stop_to_stop_id": od_id,
                "mean_error": mean_error,
                "abs_mean_error": abs(mean_error),
                "lower": mean_error,
                "upper": mean_error,
                "ci_width": 0.0,
                "n_samples": n,
                "mean_target": mean_target,
                "mean_prediction": mean_prediction
            })
            continue

        means = [rng.choice(errors, size=n, replace=True).mean() for _ in range(n_boot)]
        lower = np.percentile(means, (100 - ci) / 2)
        upper = np.percentile(means, 100 - (100 - ci) / 2)
        mean_error = np.mean(errors)

        od_stats.append({
            "stop_to_stop_id": od_id,
            "mean_error": mean_error,
            "abs_mean_error": abs(mean_error),
            "lower": lower,
            "upper": upper,
            "ci_width": upper - lower,
            "n_samples": n,
            "mean_target": mean_target,
            "mean_prediction": mean_prediction
        })

    df_out = pd.DataFrame(od_stats).sort_values("mean_error").reset_index(drop=True)
    return df_out


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    # dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR, cfg)
    # seq_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_seq)
    # lstm_input_dim = next(iter(seq_route_lookup.values())).shape[1]
    # ff_input_dim = dataset_bundle.train.x.shape[1] - 3
    # model = LSTMFeedforwardCombination(cfg, lstm_input_dim, ff_input_dim)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # model.load_state_dict(torch.load("outputs/2025-10-04/23-17-34/LSTM.pth"))
    # train_loader, val_loader, test_loader = create_dataloaders(cfg, dataset_bundle, seq_route_lookup,
    #                                                            is_route_sequence=True, num_workers=4)
    # (mae, mape, rmse), abs_accuracies, relative_accuracies, test_id_targets, raw_scores, _ = evaluate(cfg, model,
    #                                                                                                   test_loader,
    #                                                                                                   device)
    # results = test_id_targets.merge(dataset_bundle.test.x[["id", "stop_to_stop_id"]], on="id", how="left")
    # results.to_parquet("results/id_targets/full_run_lstm_ids.parquet")
    results = pd.read_parquet("results/id_targets/full_run_lstm_ids.parquet")
    od_boot = bootstrap_od_errors_per_route(results)

    top_negative = od_boot.nsmallest(20, "mean_error")
    top_positive = od_boot.nlargest(20, "mean_error")
    top_accurate = od_boot.nsmallest(20, "abs_mean_error")
    top_accurate_low = od_boot[od_boot["mean_target"]< 20].nsmallest(10, "abs_mean_error")

    print("\n=== Most Accurate (closest to 0%) ===")
    print(top_accurate[[
        "stop_to_stop_id", "mean_error", "lower", "upper",
        "mean_target", "mean_prediction", "n_samples"
    ]])

    print("\n=== Most Accurate Low (closest to 0%) ===")
    print(top_accurate_low[[
        "stop_to_stop_id", "mean_error", "lower", "upper",
        "mean_target", "mean_prediction", "n_samples"
    ]])

    print("\n=== Most Underestimated (positive mean error) ===")
    print(top_positive[[
        "stop_to_stop_id", "mean_error", "lower", "upper",
        "mean_target", "mean_prediction", "n_samples"
    ]])

    print("\n=== Most Overestimated (negative mean error) ===")
    print(top_negative[[
        "stop_to_stop_id", "mean_error", "lower", "upper",
        "mean_target", "mean_prediction", "n_samples"
    ]])


if __name__ == "__main__":
    main()
