import os
from collections import defaultdict

import hydra
import torch
import folium
from branca.colormap import linear
from folium.plugins import HeatMap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import ttest_rel, wilcoxon
from shapely import wkt, MultiLineString
from sklearn.utils import resample
from tqdm import tqdm

import config.paths as paths
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
import pandas as pd
import geopandas as gpd

from config.config import Config
from data.data_conversions import load_route_lookup, iqr_filter
from data.data_processing import create_dataloaders
from data.dataset_bundle import DatasetBundle
from data.mapping_dataset import aggr_collate_fn
from model.mlp import MLP
from plot.analysis import validation_analysis
from plot.plot import plot_error_histogram, plot_error_per_target_size, plot_deviation, plot_losses, \
    plot_multiple_losses
from train.eval import evaluate

def select_metadata(cfg: Config):
    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR)

    full_df = pd.read_csv(paths.DATASETS_DIR + cfg.dataset.metadata + ".csv")
    val_metadata = full_df[full_df["id"].isin(dataset_bundle.val.x["id"])]
    val_metadata.to_parquet(paths.DATASETS_DIR + cfg.dataset.metadata + "_val.parquet")

    full_df = pd.read_csv(paths.DATASETS_DIR + cfg.dataset.geoms + ".csv")
    val_geoms = full_df[full_df["route_seq_hash"].isin(dataset_bundle.val.x["route_seq_hash"])]
    val_geoms.to_parquet(paths.DATASETS_DIR + cfg.dataset.geoms + "_val.parquet")


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
    (mae, _, _), _, _, id_targets, _ = evaluate(cfg, model, val_loader, device)
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

    route_df = geom_df.merge(results_df, on="geom_id", how="inner")
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


def show_distribution_outlier(path, factor=1.5):
    # df = pd.read_parquet(path + ".parquet")
    # df = df[df.groupby("route_seq_hash")["route_seq_hash"].transform("count") >= 4]
    #
    # df_filtered = df.groupby("route_seq_hash", group_keys=False).apply(iqr_filter, factor=factor, include_groups=True)
    # df["mean_elapsed_time"] = df.groupby("route_seq_hash")["recorded_elapsed_time"].transform("mean")
    # df["std"] = df.groupby("route_seq_hash")["recorded_elapsed_time"].transform("std")
    # df["deviation_from_mean"] = df["recorded_elapsed_time"] - df["mean_elapsed_time"]
    #
    # df_filtered["mean_elapsed_time"] = df_filtered.groupby("route_seq_hash")["recorded_elapsed_time"].transform("mean")
    # df_filtered["std"] = df_filtered.groupby("route_seq_hash")["recorded_elapsed_time"].transform("std")
    # df_filtered["deviation_from_mean"] = df_filtered["recorded_elapsed_time"] - df_filtered["mean_elapsed_time"]
    #
    # df.to_parquet(paths.RESULTS_DIR + "time_std_dev.parquet")
    # df_filtered.to_parquet(paths.RESULTS_DIR + f"time_f{factor}_std_dev.parquet")

    df = pd.read_parquet(paths.RESULTS_DIR + "time_std_dev.parquet")
    df_filtered = pd.read_parquet(paths.RESULTS_DIR + f"time_f{factor}_std_dev.parquet")
    df["zscore"] = (df["recorded_elapsed_time"] - df["mean_elapsed_time"]) / df["std"]
    df_filtered["zscore"] = (df_filtered["recorded_elapsed_time"] - df_filtered["mean_elapsed_time"]) / df_filtered[
        "std"]

    plot_deviation(df, df_filtered, 0.963 , log_scale=False)
    plot_deviation(df, df_filtered, 0.963, log_scale=True)

def bootstrap_ci(values: pd.DataFrame, model_name, n_boot=10_000, ci=95, seed=42):
    results = {}
    rng = np.random.default_rng(seed)

    result_string = model_name

    for metric in ["MAE", "MAPE", "RMSE"]:
        arr = values[metric].to_numpy()
        n = len(arr)
        means = []
        for _ in tqdm(range(n_boot), desc=f"Bootstrapping {metric}"):
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

    print("Paired t-test p =", p_val_t)
    print("Wilcoxon p =", res.pvalue, res.statistic, res)

def get_od_results(results):
    def metrics(df):
        errors = df["prediction"] - df["target"]
        abs_errors = errors.abs()
        mae = abs_errors.mean()

        mape = (abs_errors / df["target"].replace(0, np.nan)).mean() * 100

        rmse = np.sqrt((errors**2).mean())
        return pd.Series({"MAE": mae, "MAPE": mape, "RMSE": rmse})

    return results.groupby("stop_to_stop_id").apply(metrics)

def get_interesting_results(cfg: Config, output_dir):

    mlp_results = pd.read_parquet(f"{output_dir}/MLP/dataset_time_id_targets.parquet")
    lstm_results = pd.read_parquet(f"{output_dir}/LSTM/dataset_time_id_targets.parquet")

    metadata = pd.read_parquet(paths.DATASETS_DIR + cfg.dataset.metadata + "_test_final.parquet")


    merged = metadata.merge(
            mlp_results[["id", "prediction", "target"]].rename(columns={"prediction": "mlp_prediction"}),
            on="id",
            how="left"
        ).merge(
            lstm_results[["id", "prediction"]].rename(columns={"prediction": "lstm_prediction"}),
            on="id",
            how="left"
        )

    merged["mlp_error_pct"] = (merged["mlp_prediction"] - merged["target"]) / merged["target"] * 100
    merged["lstm_error_pct"] = (merged["lstm_prediction"] - merged["target"]) / merged["target"] * 100

    db = DatasetBundle.load(paths.DATASET_BUNDLE_DIR, use_validation=False)
    merged = merged.merge(db.test.x, how="left", on="id")

    merged.sort_values("lstm_error_pct", ascending=False).head(10000).to_parquet(paths.RESULTS_DIR + f"lstm_sort.parquet")
    merged.sort_values("lstm_error_pct", ascending=True).head(10000).to_parquet(
        paths.RESULTS_DIR + f"lstm_sort_neg.parquet")

@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):

    output_dir = "outputs/2025-09-20/17-06-23"
    get_interesting_results(cfg, output_dir)
    return
    # id_targets = pd.read_parquet('outputs/2025-09-20/17-06-23/LSTM/dataset_time_id_targets.parquet')
    # model_dir = "outputs/2025-09-20/17-06-23/LSTM/new/"
    # split = "test"
    # use_subset = False
    # validation_analysis(id_targets, model_dir, split, use_subset)
    # return
    # dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR)
    # ids = dataset_bundle.test.x["id"]
    # ids.to_csv(paths.RESULTS_DIR + "test_ids.csv")
    # return
    # if cfg.dataset.use_subset:
    #     dir = "results/pca_run/"
    # else:
    #     dir = "outputs/2025-09-06/14-44-34/"
    #
    # db = DatasetBundle.load(paths.DATASET_BUNDLE_DIR, use_validation=False)
    # mlp_results = pd.read_parquet("outputs/2025-09-20/14-40-36/MLP/dataset_time_id_targets.parquet")
    # mlp_results = mlp_results.merge(db.test.x[["id", "stop_to_stop_id"]], on="id", how="left")
    # lstm_results = pd.read_parquet("outputs/2025-09-20/14-40-36/LSTM/dataset_time_id_targets.parquet")
    # lstm_results = lstm_results.merge(db.test.x[["id", "stop_to_stop_id"]], on="id", how="left")
    #
    # mlp_od = get_od_results(mlp_results)
    # lstm_od = get_od_results(lstm_results)
    #
    # mlp_bootstrap, mlp_res_string = bootstrap_ci(mlp_od, model_name="MLP")
    # lstm_bootstrap, lstm_res_string = bootstrap_ci(lstm_od, model_name="LSTM")
    # print(mlp_res_string)
    # print(lstm_res_string)
    #
    # paired_significance_test(mlp_od["MAE"], lstm_od["MAE"])
    # return

    # df = pd.read_parquet(paths.DATASETS_DIR + cfg.dataset.time + ".parquet")
    # # scores_boxplot(id_targets_dict, output_dir=dir)
    # mlp_results["mlp_prediction"] = mlp_results["prediction"]
    #
    # metadata = pd.read_parquet(paths.DATASETS_DIR + cfg.dataset.metadata + "_test_final.parquet")
    #
    #
    # merged = metadata.merge(
    #     mlp_results[["id", "prediction", "target"]].rename(columns={"prediction": "mlp_prediction"}),
    #     on="id",
    #     how="left"
    # ).merge(
    #     lstm_results[["id", "prediction"]].rename(columns={"prediction": "lstm_prediction"}),
    #     on="id",
    #     how="left"
    # )
    #
    # # Percentage errors berekenen (MAPE per sample)
    # merged["mlp_error_pct"] = (merged["mlp_prediction"] - merged["target"]).abs() / merged["target"] * 100
    # merged["lstm_error_pct"] = (merged["lstm_prediction"] - merged["target"]).abs() / merged["target"] * 100
    #
    # # Verschil tussen modellen
    # merged["prediction_diff"] = (merged["mlp_prediction"] - merged["lstm_prediction"]).abs()
    # merged["error_diff"] = (merged["mlp_error_pct"] - merged["lstm_error_pct"]).abs()
    #
    # merged = merged[(merged["mlp_error_pct"] < 200) & (merged["lstm_error_pct"] < 200)]
    # # Sorteren op grootste verschil
    # merged = merged.sort_values("error_diff", ascending=False)
    # dir = paths.RESULTS_DIR + "/analysis/"
    # os.makedirs(dir, exist_ok=True)
    # merged.to_parquet(dir + "prediction_diff.parquet")

    # df_filtered = df[df["id"].isin(id_targets["id"])]
    # df_filtered.to_parquet(paths.DATASETS_DIR + cfg.dataset.time + "_test_final.parquet")
    #
    # full_df = pd.read_csv(paths.DATASETS_DIR + cfg.dataset.metadata + ".csv")
    # test_metadata = full_df[full_df["id"].isin(id_targets["id"])]
    # test_metadata.to_parquet(paths.DATASETS_DIR + cfg.dataset.metadata + "_test_final.parquet")
    #
    # geoms = pd.read_csv(paths.DATASETS_DIR + cfg.dataset.geoms + ".csv")
    # test_geoms = geoms[geoms["geom_id"].isin(test_metadata["geom_id"])]
    # test_geoms.to_parquet(paths.DATASETS_DIR + cfg.dataset.geoms + "_test_final.parquet")
    # metadata_df = pd.read_parquet(paths.DATASETS_DIR + cfg.dataset.metadata + "_test_final.parquet")

    # for model, id_targets in id_targets_dict.items():
    # model = "MLP"
    #
    # model_dir = f"{dir}/{model}/"

    id_targets.sort_values("error", ascending=False, inplace=True)
    id_targets["abs_error"] = id_targets["error"].abs()
    merged.sort_values("abs_error", ascending=False, inplace=True)
    # print("hi")
    # neg_error = test_df[test_df["error"] < 0]
    # neg_error.head(100).to_parquet(paths.RESULTS_DIR + "neg_error.parquet")
    # merged.to_parquet(paths.RESULTS_DIR + "test_output.parquet", index=False)
    # route = test_df[test_df["route_seq_hash"] == "be0c55ce7b0ec0aed631a1a676132dee"]
    # route.to_parquet(paths.RESULTS_DIR + "route.parquet")




    # train_list = []
    # val_list = []
    # dir = "outputs/2025-08-18/13-22-53/losses/"
    # for i in range(0, 8):
    #     train_losses = np.load(dir + f"train_{i}.npy")
    #     train_list.append(train_losses)
    #     val_losses = np.load(dir + f"val_{i}.npy")
    #     val_list.append(val_losses)
    # plot_multiple_losses(train_list, val_list)
    # show_distribution_outlier(paths.DATASETS_DIR + cfg.dataset.time)
    # id_targets = pd.read_parquet(paths.RESULTS_DIR + "MLP_dataset_time_id_targets.parquet")
    # validation_analysis(id_targets)
    # select_metadata(cfg)
    # print_large_errors()
    # heatmap_per_hour_block()

if __name__ == '__main__':
    main()
