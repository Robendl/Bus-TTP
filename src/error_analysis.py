import pandas as pd
import torch.multiprocessing as mp
from tqdm import tqdm

mp.set_start_method("spawn", force=True)
import hydra
import numpy as np

from config.config import Config
import config.paths as paths

import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["HYDRA_FULL_ERROR"] = "1"

def bootstrap_od_errors_per_route(results: pd.DataFrame, n_boot=10, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    od_groups = results.groupby("stop_to_stop_id")
    print(od_groups.get_group("NL:Q:31001067NL:Q:31001079"))
    od_stats = []

    for od_id, df in tqdm(od_groups, desc="Bootstrapping OD errors"):
        errors = ((df["prediction"].to_numpy() - df["target"].to_numpy()) / df["target"].replace(0, np.nan).to_numpy()) * 100
        n = len(errors)
        if n < 2:
            mean_error = errors.mean()
            od_stats.append({
                "stop_to_stop_id": od_id,
                "mean_error": mean_error,
                "lower": mean_error,
                "upper": mean_error,
                "ci_width": 0.0,
                "n_samples": n
            })
            continue

        means = [rng.choice(errors, size=n, replace=True).mean() for _ in range(n_boot)]
        lower = np.percentile(means, (100 - ci) / 2)
        upper = np.percentile(means, 100 - (100 - ci) / 2)
        mean_error = np.mean(errors)

        od_stats.append({
            "stop_to_stop_id": od_id,
            "mean_error": mean_error,
            "lower": lower,
            "upper": upper,
            "ci_width": upper - lower,
            "n_samples": n
        })

    df_out = pd.DataFrame(od_stats).sort_values("mean_error").reset_index(drop=True)
    return df_out

@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    results = pd.read_parquet("outputs/2025-09-23/13-45-45/LSTM/id_targets.parquet")
    od_boot = bootstrap_od_errors_per_route(results)

    top_negative = od_boot.nsmallest(10, "mean_error")

    # Routes met structurele onderschatting (positieve fout)
    top_positive = od_boot.nlargest(10, "mean_error")

    print(top_negative)
    print(top_positive)


if __name__ == "__main__":
    main()
