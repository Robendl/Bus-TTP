import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from hydra.core.hydra_config import HydraConfig
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ttest_rel, wilcoxon
from tqdm import tqdm
from shapely import wkt

import contextily as cx

import config.paths as paths
from config.config import Config
from plot.plot import plot_error_per_target_size, plot_error_histogram

def plot_single_heatmap(results_df: pd.DataFrame, model_dir, split):
    results_df = results_df.groupby("geom_id")["error"].mean().reset_index()

    geom_df = pd.read_parquet(paths.DATASETS_DIR + f"dataset_geoms_{split}.parquet")
    geom_df["geom"] = geom_df["merged_geom"].apply(wkt.loads)
    route_df = geom_df.merge(results_df, on="geom_id", how="inner")
    route_df = gpd.GeoDataFrame(route_df, geometry="geom", crs="EPSG:4326").to_crs(epsg=3857)

    vmin, vmax = -100, 100
    vcenter = 0
    cmap = cm.get_cmap("managua").copy()
    cmap.set_under("green")
    cmap.set_over("#00FF00")
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 12))
    route_df.plot(
        column="error",
        cmap=cmap,
        norm=norm,
        linewidth=3,
        legend=False,
        ax=ax
    )
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
    plt.clf()
    plt.close()


def plot_heatmap(results_df: pd.DataFrame, model_dir, split, type: str):
    results_df = results_df.groupby(["geom_id", "group"])["error"].mean().reset_index()
    geom_df = pd.read_parquet(paths.DATASETS_DIR + f"dataset_geoms_{split}.parquet")
    geom_df["geom"] = geom_df["merged_geom"].apply(wkt.loads)
    route_df = geom_df.merge(results_df, on="geom_id", how="inner")
    route_df = gpd.GeoDataFrame(route_df, geometry="geom", crs="EPSG:4326").to_crs(epsg=3857)

    blocks = sorted(route_df["group"].unique())
    n_blocks = len(blocks)

    ncols = 3
    nrows = (n_blocks + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

    # Flatten axes for iteration
    axes = axes.flatten()

    vmin, vmax = -100, 100
    vcenter = 0
    xmin, ymin, xmax, ymax = route_df.total_bounds
    cmap = cm.get_cmap("coolwarm").copy()
    cmap.set_under("green")
    cmap.set_over("yellow")
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    for i, block in tqdm(enumerate(blocks), total=len(blocks)):
        ax = axes[i]
        subset = route_df[route_df["group"] == block]
        subset.plot(
            column="error",
            cmap=cmap,
            norm=norm,
            linewidth=2,
            legend=False,
            ax=ax
        )
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
    plt.savefig(f'{model_dir}/heatmap_{type}.png', dpi=300)
    plt.clf()
    plt.close()

def bootstrap_ci(values: pd.DataFrame, model_name, seed, n_boot=10_000, ci=95):
    results = {}
    rng = np.random.default_rng(seed)

    result_string = model_name

    for metric in ["MAE", "MAPE", "RMSE"]:
        arr = values[metric].to_numpy()
        n = len(arr)
        means = []
        for _ in range(n_boot):
            sample = rng.choice(arr, size=n, replace=True)
            means.append(sample.mean())
        lower = np.percentile(means, (100 - ci) / 2)
        upper = np.percentile(means, 100 - (100 - ci) / 2)
        result_string += f" & {arr.mean():.2f} [{lower:.2f}, {upper:.2f}]"
        results[metric] = {
            "mean": arr.mean(),
            "lower": lower,
            "upper": upper,
        }
    return results, result_string

def paired_significance_test(errors1, errors2):
    diffs = errors1 - errors2
    t_stat, p_val_t = ttest_rel(errors1, errors2)
    res = wilcoxon(x=errors1, y=errors2)
    return p_val_t, res.pvalue

def get_od_results(results):
    def metrics(df):
        errors = df["prediction"] - df["target"]
        abs_errors = errors.abs()
        mae = abs_errors.mean()

        mape = (abs_errors / df["target"].replace(0, np.nan)).mean() * 100

        rmse = np.sqrt((errors**2).mean())
        return pd.Series({"MAE": mae, "MAPE": mape, "RMSE": rmse})

    return results.groupby("stop_to_stop_id").apply(metrics)

def residual_plots(cfg: Config, id_targets: pd.DataFrame, model_dir, split, use_subset, relative=True):
    test_unscaled = pd.read_parquet(paths.DATASETS_DIR + cfg.dataset.time + "_test.parquet")
    df = id_targets.merge(test_unscaled, on="id", how="left")
    route_unscaled = pd.read_csv(paths.DATASETS_DIR + cfg.dataset.route_aggr + ".csv")
    residual_route_features = [
        f for f in cfg.dataset.residual_plot_features
        if f in cfg.dataset.route_feature_names
    ]
    df = df.merge(route_unscaled[["route_seq_hash"] + residual_route_features], on="route_seq_hash", how="left")
    if relative:
        df["residual"] = (df["target"] - df["prediction"]) / df["target"] * 100
    else:
        df["residual"] = (df["target"] - df["prediction"])
    n_features = len(cfg.dataset.residual_plot_features)
    n_cols = 1
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    sample_size = 50000
    sample = df.sample(n=sample_size, random_state=cfg.training.random_state) if len(df) > sample_size else df
    theta = np.arctan2(sample["sin_time"].values, sample["cos_time"].values)
    sample["time"] = ((theta % (2 * np.pi)) / (2 * np.pi)) * 24

    for i, feature in enumerate(cfg.dataset.residual_plot_features):
        feature_to_plot = feature
        ax = axes.flat[i]
        sns.scatterplot(
            data=sample,
            x=feature_to_plot,
            y="residual",
            alpha=0.3,
            ax=ax
        )
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel(feature_to_plot.replace("_", " ").capitalize())
        if relative:
            ax.set_ylabel("Relative residual (%)")
        else:
            ax.set_ylabel("Residual")

        plt.figure()
        sns.scatterplot(
            data=sample,
            x=feature_to_plot,
            y="residual",
            alpha=0.3
        )
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel(feature_to_plot.replace("_", " ").capitalize())
        if relative:
            plt.ylabel("Relative residual (%)")
        else:
            plt.ylabel("Residual")
        plt.savefig(f"{model_dir}/residual_{"rel" if relative else "abs"}_{feature_to_plot}.png", bbox_inches="tight")
        plt.close()

    for j in range(i + 1, len(axes.flat)):
        fig.delaxes(axes.flat[j])

    fig.tight_layout()
    fig.savefig(f"{model_dir}/residual_{"rel" if relative else "abs"}_all.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def validation_analysis(id_targets: pd.DataFrame, model_dir, split, use_subset):
    results_df = id_targets
    results_df["error"] = ((results_df["prediction"] - results_df["target"]) / results_df["target"]) * 100
    results_df["abs_error"] = results_df["error"].abs()
    plot_error_per_target_size(results_df.copy(), model_dir)
    plot_error_histogram(results_df["error"].copy(), model_dir)
    if not use_subset:
        metadata = pd.read_parquet(paths.DATASETS_DIR + f"dataset_metadata_{split}.parquet")
        results_df = results_df.merge(metadata, on="id")
        plot_single_heatmap(results_df, model_dir, split)
        results_df["recordeddeparturetime"] = pd.to_datetime(results_df["recordeddeparturetime"], format='mixed')
        results_df["group"] = (results_df["recordeddeparturetime"].dt.hour // 4) * 4
        plot_heatmap(results_df, model_dir, split, type="hour")
        results_df["group"] = results_df["recordeddeparturetime"].dt.month
        plot_heatmap(results_df, model_dir, split, type="month")

def high_error_examples(cfg: Config, id_targets, output_dir):

    metadata = pd.read_parquet(paths.DATASETS_DIR + cfg.dataset.metadata + "_test_final.parquet")

    merged = metadata.merge(id_targets, on="id", how="left")

    merged["mlp_error_pct"] = (merged["mlp_prediction"] - merged["target"]) / merged["target"] * 100

    test_df = pd.read_parquet(paths.DATASETS_DIR + cfg.dataset.time + "_test.parquet")
    merged = merged.merge(test_df, how="left", on="id")

    dir = paths.RESULTS_DIR + "/error_analysis/"
    os.makedirs(dir, exist_ok=True)

    merged.sort_values("lstm_error_pct", ascending=False).head(10000).to_parquet(dir + f"lstm_sort_desc.parquet")
    merged.sort_values("lstm_error_pct", ascending=True).head(10000).to_parquet(dir + f"lstm_sort_asc.parquet")
    merged.sort_values("mlp_error_pct", ascending=False).head(10000).to_parquet(dir + f"mlp_sort_desc.parquet")
    merged.sort_values("mlp_error_pct", ascending=True).head(10000).to_parquet(dir + f"mlp_sort_asc.parquet")

    merged["prediction_diff"] = (merged["mlp_prediction"] - merged["lstm_prediction"]).abs()
    merged["error_diff"] = (merged["mlp_error_pct"] - merged["lstm_error_pct"]).abs()

    merged.sort_values("error_diff", ascending=False).head(1000).to_parquet(dir + f"diff_error_sort.parquet")

if __name__ == '__main__':
    id_targets = pd.read_parquet('outputs/2025-09-22/13-21-12/MLP/dataset_time_id_targets.parquet')
    model_dir = "results/more_filtering/LSTM/new/"
    split = "test"
    use_subset = False
    validation_analysis(id_targets, model_dir, split, use_subset)