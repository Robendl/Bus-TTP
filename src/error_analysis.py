"""Per OD-pair error diagnostics with bootstrap confidence intervals."""
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm

import config.paths as paths
from config.config import Config
from runtime import setup_environment

setup_environment()


def bootstrap_od_errors_per_route(
    results: pd.DataFrame,
    n_boot: int = 100,
    ci: int = 95,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute per-OD-pair mean percentage error with bootstrap confidence intervals."""
    rng = np.random.default_rng(seed)
    od_stats = []

    for od_id, df in tqdm(results.groupby("stop_to_stop_id"), desc="Bootstrapping OD errors"):
        errors = ((df["prediction"].to_numpy() - df["target"].to_numpy())
                  / df["target"].replace(0, np.nan).to_numpy()) * 100
        errors = errors[~np.isnan(errors)]
        n = len(errors)
        mean_target = df["target"].mean()
        mean_prediction = df["prediction"].mean()

        if n < 2:
            mean_error = float(np.nanmean(errors))
            od_stats.append({
                "stop_to_stop_id": od_id,
                "mean_error": mean_error,
                "abs_mean_error": abs(mean_error),
                "lower": mean_error,
                "upper": mean_error,
                "ci_width": 0.0,
                "n_samples": n,
                "mean_target": mean_target,
                "mean_prediction": mean_prediction,
            })
            continue

        boot_means = [rng.choice(errors, size=n, replace=True).mean() for _ in range(n_boot)]
        lower = float(np.percentile(boot_means, (100 - ci) / 2))
        upper = float(np.percentile(boot_means, 100 - (100 - ci) / 2))
        mean_error = float(np.mean(errors))

        od_stats.append({
            "stop_to_stop_id": od_id,
            "mean_error": mean_error,
            "abs_mean_error": abs(mean_error),
            "lower": lower,
            "upper": upper,
            "ci_width": upper - lower,
            "n_samples": n,
            "mean_target": mean_target,
            "mean_prediction": mean_prediction,
        })

    return pd.DataFrame(od_stats).sort_values("mean_error").reset_index(drop=True)


_DISPLAY_COLS = [
    "stop_to_stop_id", "mean_error", "lower", "upper",
    "mean_target", "mean_prediction", "n_samples",
]


def _print_section(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(df[_DISPLAY_COLS])


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    results_path = Path("results/id_targets/full_run_lstm_ids.parquet")
    results = pd.read_parquet(results_path)
    od_boot = bootstrap_od_errors_per_route(results, seed=cfg.training.random_state)

    _print_section("Most accurate (closest to 0%)", od_boot.nsmallest(20, "abs_mean_error"))
    _print_section(
        "Most accurate among short trips (mean target < 20s)",
        od_boot[od_boot["mean_target"] < 20].nsmallest(10, "abs_mean_error"),
    )
    _print_section("Most underestimated (positive mean error)", od_boot.nlargest(20, "mean_error"))
    _print_section("Most overestimated (negative mean error)", od_boot.nsmallest(20, "mean_error"))


if __name__ == "__main__":
    main()
