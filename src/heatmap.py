from collections import defaultdict

import hydra
import torch
import folium
from branca.colormap import linear
from folium.plugins import HeatMap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from shapely import wkt, MultiLineString
from tqdm import tqdm

import config.paths as paths
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
import pandas as pd
import geopandas as gpd

from config.config import Config
from data.data_conversions import load_route_lookup
from data.data_processing import create_dataloaders
from data.dataset_bundle import DatasetBundle
from data.mapping_dataset import aggr_collate_fn
from model.mlp import MLP
from plot.plot import plot_error_histogram, plot_error_per_target_size
from train.eval import evaluate

def add_geometries(cfg: Config):
    results = pd.read_parquet(paths.RESULTS_DIR + "result_analysis.parquet")
    metadata = pd.read_csv(paths.DATASETS_DIR + "dataset_metadata.csv")
    # metadata.to_parquet(paths.DATASETS_DIR + "dataset_metadata.parquet")
    print(results.shape, flush=True)
    results = results.merge(metadata[['id', 'geom_id']], on="id", how="left")
    results.to_parquet(paths.RESULTS_DIR + "results_analysis.parquet")
    print(results.shape, flush=True)

    geoms = pd.read_csv(paths.DATASETS_DIR + "dataset_geoms.csv")
    unique_ids = results['geom_id'].unique()
    print(geoms.shape, flush=True)
    geoms = geoms[geoms['geom_id'].isin(unique_ids)]
    print(geoms.shape, flush=True)
    geoms["geom"] = geoms["merged_geom"].apply(wkt.loads)

    print("creating geodataframe", flush=True)
    gdf = gpd.GeoDataFrame(geoms, geometry="geom", crs="EPSG:4326")
    print("saving geodataframe", flush=True)
    gdf.to_parquet(paths.RESULTS_DIR + "results_geo.parquet")

def get_scores(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR)
    aggr_route_lookup = load_route_lookup(paths.DATASETS_DIR + cfg.dataset.route_aggr)

    model = MLP(cfg.model.input_dim, cfg.model.mlp.hidden_dims, cfg.model.output_dim)
    model.load_state_dict(torch.load("/home3/s3799174/bus-travel-time-prediction/outputs/2025-07-21/19-17-36/MLP.pth"))
    model.to(device)
    train_loader, val_loader, test_loader = create_dataloaders(cfg, dataset_bundle, aggr_route_lookup,
                                                               aggr_collate_fn, num_workers=4)
    mae, _, _, id_targets = evaluate(cfg, model, val_loader, device)
    metadata = pd.read_parquet(paths.DATASETS_DIR + "dataset_metadata.parquet")
    merged_df = id_targets.merge(metadata, on="id")
    print(merged_df.shape)
    print(merged_df.head)
    merged_df.to_parquet(paths.RESULTS_DIR + "result_analysis.parquet")

def explode_geometry_to_points(row):
    geom = row["geom"]
    error = abs(row.prediction - row.target)

    # Zorg voor uniforme lijst van lijnen
    lines = geom.geoms if isinstance(geom, MultiLineString) else [geom]

    points = []
    for line in lines:
        for lon, lat in line.coords:
            points.append({
                "lat": lat,
                "lon": lon,
                "error": error,
                "geom_id": row.geom_id,
                "prediction": row.prediction,
                "target": row.target
            })
    return points

def geometry_to_points(geom):
    lines = geom.geoms if isinstance(geom, MultiLineString) else [geom]
    return [(pt[1], pt[0]) for line in lines for pt in line.coords]  # lat, lon


def heatmap(results_df, geom_df):
    geom_points = {
        row.geom_id: geometry_to_points(row["geom"])
        for _, row in tqdm(geom_df.iterrows(), total=geom_df.shape[0])
    }

    all_points = []
    cell_errors = defaultdict(list)

    for _, row in tqdm(results_df.iterrows(), total=results_df.shape[0]):
        error = abs(row.prediction - row.target)
        points = geom_points.get(row.geom_id, [])

        for lat, lon in points:
            # Maak bin (bijv. 0.01 ≈ ~1 km, 0.005 ≈ ~500 m, 0.001 ≈ ~100 m)
            lat_bin = round(lat, 3)
            lon_bin = round(lon, 3)
            cell_errors[(lat_bin, lon_bin)].append(error)

    heatmap_data = [
        {"lat": lat, "lon": lon, "avg_error": sum(errs) / len(errs)}
        for (lat, lon), errs in cell_errors.items()
    ]

    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.to_parquet(paths.RESULTS_DIR + "heatmap_df.parquet")
    print("creating heatmap", flush=True)
    m = folium.Map(location=[heatmap_df["lat"].mean(), heatmap_df["lon"].mean()], zoom_start=12)

    heat_data = heatmap_df[["lat", "lon", "avg_error"]].values.tolist()
    HeatMap(heat_data, radius=10).add_to(m)

    m.save(paths.RESULTS_DIR + "avg_error_heatmap.html")


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def heatmap_per_hour_block(cfg: Config):
    results_df = pd.read_parquet(paths.RESULTS_DIR + "results_analysis.parquet")
    geom_df = gpd.read_parquet(paths.RESULTS_DIR + "results_geo.parquet")
    results_df = results_df[results_df["target"] > 0]
    results_df["error"] = ((results_df["prediction"] - results_df["target"]) / results_df["target"]) * 100
    results_df["abs_error"] = results_df["error"].abs()
    results_df = results_df[(results_df["error"] > -100) & (results_df["error"] < 200)]
    plot_error_per_target_size(results_df.copy())
    plot_error_histogram(results_df["error"])
    results_df["recordeddeparturetime"] = pd.to_datetime(results_df["recordeddeparturetime"], format='mixed')
    results_df["hour_group"] = (results_df["recordeddeparturetime"].dt.hour // 4) * 4
    print("hour", flush=True)
    results_df = results_df.groupby(["geom_id", "hour_group"])["error"].mean().reset_index()
    # print("grouped", flush=True)
    # results_df = results_df.merge(geom_df[['geom_id', 'geom']], on="geom_id", how="left")
    # print("merged", flush=True)
    # results_gdf = gpd.GeoDataFrame(results_df, geometry="geom", crs="EPSG:4326")
    # print("saving", flush=True)
    # results_gdf.to_file(paths.RESULTS_DIR + "results.geojson", driver="GeoJSON")
    # return

    route_df = geom_df.merge(results_df, on="geom_id", how="inner")  # of left als je nulls wilt behouden
    route_df = gpd.GeoDataFrame(route_df, geometry="geom", crs="EPSG:4326").to_crs(epsg=3857)

    # Stap 4: subplots voorbereiden
    hour_blocks = sorted(route_df["hour_group"].unique())
    n_blocks = len(hour_blocks)

    ncols = 3
    nrows = (n_blocks + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

    # Flatten axes for iteration (werkt ook als laatste subplot leeg is)
    axes = axes.flatten()

    vmin, vmax = -200, 200
    xmin, ymin, xmax, ymax = route_df.total_bounds
    cmap = "coolwarm"
    norm = Normalize(vmin=vmin, vmax=vmax)

    for i, hour in tqdm(enumerate(hour_blocks), total=len(hour_blocks)):
        ax = axes[i]
        subset = route_df[route_df["hour_group"] == hour]
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
        ax.set_title(f"{str(hour).zfill(2)}:00–{str(hour + 3).zfill(2)}:59")
        ax.axis("off")

    plt.tight_layout(rect=(0, 0.08, 1, 1))
    # 7. Eén centrale legend (colorbar)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar_ax = fig.add_axes((0.25, 0.05, 0.5, 0.02))  # midden, onderin
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Mean Error (s)")

    plt.savefig(paths.RESULTS_DIR + "heatmap_relative_errors.png", dpi=300)



    # plot_error_histogram(results_df["error"].to_numpy())
    #
    # avg_errors = results_df.groupby("geom_id")["error"].mean().reset_index()
    #
    # route_df = geom_df.merge(avg_errors, on="geom_id")
    # route_df = route_df.to_crs(epsg=3857)
    #
    # ax = route_df.plot(
    #     column="error",
    #     cmap="coolwarm",
    #     linewidth=2,
    #     legend=True,
    #     figsize=(12, 10)
    # )
    # cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    #
    # # route_df.plot(column="error", cmap="YlOrRd", linewidth=1.5, legend=True, ax=ax)
    # plt.title("Average error per route")
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig(paths.RESULTS_DIR + "routes_by_error.png", dpi=300)



    # m = folium.Map(location=[52.37, 4.89], zoom_start=11)
    #
    # colormap = linear.YlOrRd_09.scale(route_df["error"].min(), route_df["error"].max())
    # colormap.caption = "Mean error per route"
    # colormap.add_to(m)
    #
    # def style_function(feature):
    #     error = feature["properties"]["error"]
    #     return {
    #         "color": colormap(error),
    #         "weight": 4,
    #         "opacity": 0.8
    #     }
    #
    # folium.GeoJson(
    #     route_df,
    #     style_function=style_function,
    #     tooltip=folium.GeoJsonTooltip(fields=["geom_id", "error"])
    # ).add_to(m)
    #
    # m.save(paths.RESULTS_DIR + "routes_by_error.html")

def print_large_errors():
    results_df = pd.read_parquet(paths.RESULTS_DIR + "results_analysis.parquet")
    geom_df = gpd.read_parquet(paths.RESULTS_DIR + "results_geo.parquet")
    results_df["error"] = ((results_df["prediction"] - results_df["target"]) / results_df["target"]) * 100
    large_errors = results_df[results_df["error"] > 200]
    print(large_errors.shape)
    large_errors = large_errors[(large_errors["target"] > 10)]
    print(large_errors)
    print(large_errors.shape)
    print(results_df.shape)

if __name__ == '__main__':
    print_large_errors()
    # heatmap_per_hour_block()