import hydra
import torch
import folium
from folium.plugins import HeatMap
from shapely import wkt

import config.paths as paths
import numpy as np
import pandas as pd
import geopandas as gpd

from config.config import Config
from data.data_conversions import load_route_lookup
from data.data_processing import create_dataloaders
from data.dataset_bundle import DatasetBundle
from data.mapping_dataset import aggr_collate_fn
from model.mlp import MLP
from train.eval import evaluate

def add_geometries(cfg: Config):
    results = pd.read_parquet(paths.RESULTS_DIR + "result_analysis.parquet")
    metadata = pd.read_csv(paths.DATASETS_DIR + "dataset_metadata.csv")
    # metadata.to_parquet(paths.DATASETS_DIR + "dataset_metadata.parquet")
    print(results.shape, flush=True)
    results = results.merge(metadata[['id', 'geom_id']], on="id", how="left")
    results.to_parquet(paths.DATASETS_DIR + "results_analysis.parquet")
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


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def heatmap(cfg: Config):
    add_geometries(cfg)
    return
    df = pd.read_parquet(paths.RESULTS_DIR + "result_analysis.parquet")
    df["abs_error"] = (df["prediction"] - df["target"]).abs()

    # Maak folium kaart, centraal op NL of gemiddelde locatie
    center_lat = df["mid_lat"].mean()
    center_lon = df["mid_lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    # Maak heatmap met (lat, lon, weight)
    df["lat_bin"] = df["mid_lat"].round(3)
    df["lon_bin"] = df["mid_lon"].round(3)

    # Gemiddelde error per (lat, lon) bin
    grouped = df.groupby(["lat_bin", "lon_bin"])["abs_error"].mean().reset_index()

    # Maak heatmap met gemiddelde error als gewicht
    heat_data = grouped[["lat_bin", "lon_bin", "abs_error"]].values.tolist()
    HeatMap(heat_data, radius=10).add_to(m)

    # Bewaar of toon kaart
    m.save(paths.RESULTS_DIR + "error_heatmap.html")


if __name__ == '__main__':
    heatmap()