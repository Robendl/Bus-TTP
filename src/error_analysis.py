import pandas as pd
import torch.multiprocessing as mp
from tqdm import tqdm
import hydra
import numpy as np
import os

from config.config import Config
import config.paths as paths

mp.set_start_method("spawn", force=True)
os.environ["WANDB_MODE"] = "disabled"
os.environ["HYDRA_FULL_ERROR"] = "1"


def bootstrap_od_errors_per_route(results: pd.DataFrame, n_boot=10, ci=95, seed=42):
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
    results = pd.read_parquet("outputs/2025-09-23/13-45-45/LSTM/id_targets.parquet")
    od_boot = bootstrap_od_errors_per_route(results)

    top_negative = od_boot.nsmallest(10, "mean_error")
    top_positive = od_boot.nlargest(10, "mean_error")
    top_accurate = od_boot.nsmallest(10, "abs_mean_error")
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
