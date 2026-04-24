"""Generic plotting utilities (loss curves, error histograms, TAC, etc.)."""
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from matplotlib import ticker
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

# Models share a single qualitative palette so they're consistent across plots.
_MODEL_PALETTE = plt.get_cmap("Set1")
_HIGHLIGHT_MARGINS = (10, 20, 30)


def plot_tac(margins, accuracies: Dict[str, np.ndarray], metric: str, output_dir: str) -> None:
    for idx, (name, values) in enumerate(accuracies.items()):
        plt.plot(margins, values, label=name, color=_MODEL_PALETTE(idx))

    plt.xlabel(f"Tolerance margin ({'%' if metric == 'p' else metric})")
    plt.ylabel("Accuracy within margin")
    plt.ylim(0, 1)
    plt.legend(frameon=True, loc="lower right")
    plt.grid(alpha=0.3, linestyle="--")
    plt.savefig(f"{output_dir}/tolerance_acc_curve_{metric}.pdf")
    plt.close()


def bootstrap_tac_per_model(
    df_dict: Dict[str, pd.DataFrame],
    margins: np.ndarray,
    seed: int,
    output_dir: str,
    ci: int = 95,
    n_boot: int = 1000,
    percentage: bool = False,
) -> None:
    """Bootstrap (over OD pairs) confidence intervals for each model's TAC curve."""
    rng = np.random.default_rng(seed)
    margins_arr = np.asarray(margins, dtype=float)
    if percentage:
        margins_arr = margins_arr / 100.0

    results = {}
    for model_name, df in df_dict.items():
        errors = np.abs(df["prediction"].values - df["target"].values)
        if percentage:
            errors = errors / df["target"].values

        df_err = pd.DataFrame({"stop_to_stop_id": df["stop_to_stop_id"].values, "error": errors})
        pair_acc = (
            df_err.groupby("stop_to_stop_id")["error"]
            .apply(lambda e: (e.values[:, None] <= margins_arr).mean(axis=0))
        )
        pair_acc = np.stack(pair_acc.values)
        n_pairs = pair_acc.shape[0]

        tac_samples = np.empty((n_boot, len(margins_arr)))
        for b in range(n_boot):
            idx = rng.integers(0, n_pairs, n_pairs)
            tac_samples[b] = pair_acc[idx].mean(axis=0)

        results[model_name] = {
            "mean": tac_samples.mean(axis=0),
            "lower": np.percentile(tac_samples, (100 - ci) / 2, axis=0),
            "upper": np.percentile(tac_samples, 100 - (100 - ci) / 2, axis=0),
        }

    for idx, (name, res) in enumerate(results.items()):
        color = _MODEL_PALETTE(idx)
        plt.plot(margins, res["mean"], label=name, color=color)
        plt.fill_between(margins, res["lower"], res["upper"], color=color, alpha=0.2)
        for m_idx, margin in enumerate(margins):
            if margin in _HIGHLIGHT_MARGINS:
                print(f"{name}, margin {margin}: {res['mean'][m_idx] * 100:.2f}%")

    plt.xlabel("Tolerance margin (%)" if percentage else "Tolerance margin (s)")
    plt.ylabel("Accuracy within margin")
    plt.ylim(0, 1)
    plt.legend(frameon=True, loc="lower right")
    plt.grid(alpha=0.3, linestyle="--")
    suffix = "tac_percentage" if percentage else "tac_absolute"
    plt.savefig(f"{output_dir}/bootstrap_{suffix}.pdf")
    plt.close()


def plot_error_histogram(errors: pd.Series, model_dir: str, baseline: bool = False) -> None:
    threshold = 300
    max_error = int(errors.max()) + 1
    capped = np.copy(errors)
    capped[capped > threshold] = threshold + 3

    plt.hist(capped, bins=100, edgecolor="black", alpha=0.7,
             color=_MODEL_PALETTE(1), density=True)
    plt.xlabel("Percentage error")
    plt.ylabel("Density")
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    plt.grid(alpha=0.3, linestyle="--")
    plt.ylim(0, 0.022)
    plt.tight_layout()
    plt.savefig(f"{model_dir}/{'bs_' if baseline else ''}error_histogram_({max_error}).pdf")
    plt.close()


def plot_error_per_target_size(df: pd.DataFrame, model_dir: str) -> None:
    max_target = int(df["target"].max())
    bins = list(range(0, 2001, 200))
    if max_target > 2000:
        bins.append(max_target)
    labels = [f"{bins[i]}–{bins[i + 1]}" for i in range(len(bins) - 1)]
    df["target_bin"] = pd.cut(df["target"], bins=bins, labels=labels, right=False)

    bin_stats = df.groupby("target_bin", observed=False)["abs_error"].agg(["mean", "std"]).reset_index()
    fractions = df["target_bin"].value_counts().reindex(labels, fill_value=0) / df.shape[0]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(bin_stats["target_bin"], bin_stats["mean"], marker="o", label="Mean % Error")
    ax1.fill_between(
        bin_stats["target_bin"].astype(str),
        bin_stats["mean"] - bin_stats["std"],
        bin_stats["mean"] + bin_stats["std"],
        alpha=0.3,
        label="±1 Std Dev",
    )
    ax1.set_ylabel("Mean Percentage Error")

    ax2 = ax1.twinx()
    ax2.plot(labels, fractions, color="red", marker="o")
    ax2.set_ylabel("Fraction of data")
    ax2.set_ylim(0, 1)

    plt.xticks(rotation=45)
    plt.xlabel("Target Bin")
    plt.title("Mean Percentage Error by Target Size")
    plt.tight_layout()
    plt.savefig(f"{model_dir}/error_target_size.pdf")
    plt.close()


def plot_losses(train_losses, val_losses, model_name: str, output_dir: str | None = None) -> None:
    plt.plot(range(1, len(train_losses) + 1), train_losses,
             label="Train", color=_MODEL_PALETTE(1), linewidth=2)

    first_val_epoch = len(train_losses) - len(val_losses)
    if val_losses:
        plt.plot(range(first_val_epoch + 1, len(train_losses) + 1), val_losses,
                 label="Validation", color=_MODEL_PALETTE(0), linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.ylim(top=40, bottom=10)
    plt.legend(frameon=True, loc="upper right")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    if output_dir is None:
        output_dir = HydraConfig.get().run.dir
    plt.savefig(f"{output_dir}/{model_name}_losses.pdf")
    plt.close()


def _calculate_zscores(df: pd.DataFrame) -> pd.Series:
    df["mean_elapsed_time"] = df.groupby("route_seq_hash")["recorded_elapsed_time"].transform("mean")
    df["std"] = df.groupby("route_seq_hash")["recorded_elapsed_time"].transform("std")
    return (df["recorded_elapsed_time"] - df["mean_elapsed_time"]) / df["std"]


def plot_deviation(
    df: pd.DataFrame,
    df_filtered: pd.DataFrame,
    new_fraction: float,
    lower: float = 0,
    upper: float = 0,
    log_scale: bool = False,
) -> None:
    """Compare per-route z-score distributions before and after IQR filtering."""
    s1 = _calculate_zscores(df)
    s2 = _calculate_zscores(df_filtered)
    if lower != 0 and upper != 0:
        s1, s2 = s1.clip(lower, upper), s2.clip(lower, upper)

    removed_pct = 100 * (1 - new_fraction)
    plt.hist(s1, bins=100, alpha=0.5, label="Before", density=True, color=_MODEL_PALETTE(0))
    plt.hist(s2, bins=100, alpha=0.5, label=f"After (-{removed_pct:.1f}%)",
             density=True, color=_MODEL_PALETTE(1))

    filename = "z-scores"
    if log_scale:
        plt.yscale("log")
        plt.ylabel("log(Density)")
        filename += "_log"
    else:
        plt.ylabel("Density")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Z-score")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()

    output_dir = HydraConfig.get().run.dir
    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.close()


def scores_boxplot(id_targets_dict: Dict[str, pd.DataFrame], output_dir: str | None = None) -> None:
    metrics = {
        "MAE": mean_absolute_error,
        "MAPE": mean_absolute_percentage_error,
        "RMSE": root_mean_squared_error,
    }
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5), sharey=False)
    for ax, (metric, _) in zip(axes, metrics.items()):
        data, labels = [], []
        for model_name, id_targets in id_targets_dict.items():
            values = mean_absolute_error(
                np.array(id_targets["target"]),
                np.array(id_targets["prediction"]),
                multioutput="raw_values",
            )
            data.append(values)
            labels.append(model_name)
        ax.boxplot(data, tick_labels=labels)
        ax.set_title(metric)
        ax.set_ylabel("Score")

    plt.tight_layout()
    if output_dir is None:
        output_dir = HydraConfig.get().run.dir
    plt.savefig(f"{output_dir}/scores_boxplot.pdf")
    plt.close()
