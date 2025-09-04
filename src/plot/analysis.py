import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from hydra.core.hydra_config import HydraConfig
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm
from shapely import wkt

import contextily as cx

import config.paths as paths
from plot.plot import plot_error_per_target_size, plot_error_histogram


def plot_heatmap(results_df: pd.DataFrame, model_dir, type: str, split):
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

    vmin, vmax = -100, 200
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


def validation_analysis(id_targets: pd.DataFrame, model_dir, split):
    metadata = pd.read_parquet(paths.DATASETS_DIR + f"dataset_metadata_{split}.parquet")
    results_df = id_targets.merge(metadata, on="id")
    results_df["error"] = ((results_df["prediction"] - results_df["target"]) / results_df["target"]) * 100
    results_df["abs_error"] = results_df["error"].abs()
    plot_error_per_target_size(results_df.copy(), model_dir)
    plot_error_histogram(results_df["error"].copy(), model_dir)
    results_df["recordeddeparturetime"] = pd.to_datetime(results_df["recordeddeparturetime"], format='mixed')
    results_df["group"] = (results_df["recordeddeparturetime"].dt.hour // 4) * 4
    plot_heatmap(results_df, model_dir, type="hour")
    results_df["group"] = results_df["recordeddeparturetime"].dt.month
    plot_heatmap(results_df, model_dir, type="month")




