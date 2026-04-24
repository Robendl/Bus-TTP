"""Geospatial heatmaps, residual diagnostics, and statistical summaries."""
from pathlib import Path
from typing import Tuple

import contextily as cx
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import ttest_rel, wilcoxon
from shapely import wkt
from tqdm import tqdm

import config.paths as paths
from config.config import Config
from plot.plot import plot_error_histogram, plot_error_per_target_size

# Asymmetric diverging colormap that emphasizes positive residuals (under-prediction).
_HEATMAP_COLORSCHEME = [
    (0.00, (0.02, 0.16, 0.47)),
    (0.25, (0.00, 0.25, 0.75)),
    (0.40, (0.20, 0.45, 1.00)),
    (0.50, (0.55, 0.55, 0.55)),
    (0.60, (1.00, 0.45, 0.20)),
    (0.80, (0.75, 0.25, 0.00)),
    (1.00, (0.47, 0.16, 0.02)),
]


def _load_route_geoms(cfg_split: str) -> gpd.GeoDataFrame:
    df = pd.read_parquet(f"{paths.DATASETS_DIR}dataset_geoms_{cfg_split}.parquet")
    df["geom"] = df["merged_geom"].apply(wkt.loads)
    return df


def _diverging_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list("blue_grey_red_maxcontrast", _HEATMAP_COLORSCHEME, N=256)
    cmap.set_under("green")
    cmap.set_over("yellow")
    return cmap


def plot_single_heatmap(results_df: pd.DataFrame, model_dir: str, split: str) -> None:
    aggregated = results_df.groupby("geom_id")["error"].mean().reset_index()
    geom_df = _load_route_geoms(split)
    route_df = geom_df.merge(aggregated, on="geom_id", how="inner")
    route_df = gpd.GeoDataFrame(route_df, geometry="geom", crs="EPSG:4326").to_crs(epsg=3857)

    cmap = _diverging_cmap()
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)

    fig, ax = plt.subplots(figsize=(12, 12))
    route_df.plot(column="error", cmap=cmap, norm=norm, linewidth=3, legend=False, ax=ax)
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution=False)

    xmin, ymin, xmax, ymax = route_df.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04, extend="max")
    cbar.set_label("Mean Error Percentage")

    plt.savefig(f"{model_dir}/heatmap_all.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(results_df: pd.DataFrame, model_dir: str, split: str, type: str) -> None:
    """Per-bin (hour-of-day or month) geospatial error heatmaps."""
    aggregated = results_df.groupby(["geom_id", "group"])["error"].mean().reset_index()
    geom_df = _load_route_geoms(split)
    route_df = geom_df.merge(aggregated, on="geom_id", how="inner")
    route_df = gpd.GeoDataFrame(route_df, geometry="geom", crs="EPSG:4326").to_crs(epsg=3857)

    blocks = sorted(route_df["group"].unique())
    ncols = 3
    nrows = (len(blocks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()

    cmap = cm.get_cmap("coolwarm").copy()
    cmap.set_under("green")
    cmap.set_over("yellow")
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    xmin, ymin, xmax, ymax = route_df.total_bounds

    for i, block in tqdm(enumerate(blocks), total=len(blocks)):
        ax = axes[i]
        subset = route_df[route_df["group"] == block]
        subset.plot(column="error", cmap=cmap, norm=norm, linewidth=2, legend=False, ax=ax)
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution=False)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if type == "hour":
            ax.set_title(f"{str(block).zfill(2)}:00–{str(block + 3).zfill(2)}:59")
        else:
            ax.set_title(pd.to_datetime(str(block), format="%m").month_name())
        ax.axis("off")

    plt.tight_layout(rect=(0, 0.08, 1, 1))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar_ax = fig.add_axes((0.25, 0.05, 0.5, 0.02))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="max")
    cbar.set_label("Mean Error Percentage")

    plt.savefig(f"{model_dir}/heatmap_{type}.png", dpi=300)
    plt.close()


def bootstrap_ci(
    values: pd.DataFrame,
    model_name: str,
    seed: int,
    n_boot: int = 10_000,
    ci: int = 95,
) -> Tuple[dict, str]:
    """Bootstrap mean confidence intervals over OD-pair scores; returns a dict and a LaTeX-style row."""
    rng = np.random.default_rng(seed)
    results = {}
    result_string = model_name

    for metric in ["MAE", "MAPE", "RMSE"]:
        arr = values[metric].to_numpy()
        n = len(arr)
        means = [rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)]
        lower = float(np.percentile(means, (100 - ci) / 2))
        upper = float(np.percentile(means, 100 - (100 - ci) / 2))
        result_string += f" & {arr.mean():.2f} [{lower:.2f}, {upper:.2f}]"
        results[metric] = {"mean": float(arr.mean()), "lower": lower, "upper": upper}

    return results, result_string


def paired_significance_test(errors1: np.ndarray, errors2: np.ndarray) -> Tuple[float, float]:
    _, p_t = ttest_rel(errors1, errors2)
    res = wilcoxon(x=errors1, y=errors2)
    return float(p_t), float(res.pvalue)


def get_od_results(results: pd.DataFrame) -> pd.DataFrame:
    def metrics(df: pd.DataFrame) -> pd.Series:
        errors = df["prediction"] - df["target"]
        abs_errors = errors.abs()
        return pd.Series({
            "MAE": abs_errors.mean(),
            "MAPE": (abs_errors / df["target"].replace(0, np.nan)).mean() * 100,
            "RMSE": np.sqrt((errors ** 2).mean()),
        })

    return results.groupby("stop_to_stop_id").apply(metrics)


def residual_plots(
    cfg: Config,
    id_targets: pd.DataFrame,
    model_dir: str,
    split: str,
    use_subset: bool,
    relative: bool = True,
) -> None:
    """Per-feature residual scatter plots (one figure per feature plus a grid)."""
    test_unscaled = pd.read_parquet(paths.DATASETS_DIR + cfg.dataset.time + "_test.parquet")
    df = id_targets.merge(test_unscaled, on="id", how="left")
    route_unscaled = pd.read_csv(paths.DATASETS_DIR + cfg.dataset.route_aggr + ".csv")

    residual_route_features = [
        f for f in cfg.dataset.residual_plot_features if f in cfg.dataset.route_feature_names
    ]
    df = df.merge(
        route_unscaled[["route_seq_hash"] + residual_route_features],
        on="route_seq_hash",
        how="left",
    )

    if relative:
        df["residual"] = (df["target"] - df["prediction"]) / df["target"] * 100
    else:
        df["residual"] = df["target"] - df["prediction"]

    sample_size = 50_000
    sample = df.sample(n=sample_size, random_state=cfg.training.random_state) if len(df) > sample_size else df
    theta = np.arctan2(sample["sin_time"].values, sample["cos_time"].values)
    sample["time"] = ((theta % (2 * np.pi)) / (2 * np.pi)) * 24

    n_features = len(cfg.dataset.residual_plot_features)
    fig, axes = plt.subplots(n_features, 1, figsize=(6, 4 * n_features))
    suffix = "rel" if relative else "abs"
    ylabel = "Relative residual (%)" if relative else "Residual"

    for i, feature in enumerate(cfg.dataset.residual_plot_features):
        ax = axes.flat[i] if n_features > 1 else axes
        sns.scatterplot(data=sample, x=feature, y="residual", alpha=0.3, ax=ax)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel(feature.replace("_", " ").capitalize())
        ax.set_ylabel(ylabel)

        plt.figure()
        sns.scatterplot(data=sample, x=feature, y="residual", alpha=0.3)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel(feature.replace("_", " ").capitalize())
        plt.ylabel(ylabel)
        plt.savefig(f"{model_dir}/residual_{suffix}_{feature}.png", bbox_inches="tight")
        plt.close()

    fig.tight_layout()
    fig.savefig(f"{model_dir}/residual_{suffix}_all.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def validation_analysis(
    id_targets: pd.DataFrame,
    model_dir: str,
    split: str,
    use_subset: bool,
) -> None:
    """End-to-end analysis bundle: error histogram + per-target-size errors + heatmaps."""
    results_df = id_targets.copy()
    results_df["error"] = (results_df["prediction"] - results_df["target"]) / results_df["target"] * 100
    results_df["abs_error"] = results_df["error"].abs()

    plot_error_per_target_size(results_df.copy(), str(model_dir))
    plot_error_histogram(results_df["error"].copy(), str(model_dir))

    if use_subset:
        return

    metadata = pd.read_parquet(paths.DATASETS_DIR + f"dataset_metadata_{split}.parquet")
    results_df = results_df.merge(metadata, on="id")
    plot_single_heatmap(results_df, str(model_dir), split)

    results_df["recordeddeparturetime"] = pd.to_datetime(
        results_df["recordeddeparturetime"], format="mixed"
    )
    results_df["group"] = (results_df["recordeddeparturetime"].dt.hour // 4) * 4
    plot_heatmap(results_df, str(model_dir), split, type="hour")
    results_df["group"] = results_df["recordeddeparturetime"].dt.month
    plot_heatmap(results_df, str(model_dir), split, type="month")
