import hydra
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from config import paths
from config.config import Config
from data.data_conversions import load_route_lookup
from data.dataset_bundle import DatasetBundle
from data.plot_distribution import plot_distribution
from feature_selection.test import correlation_analysis


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    db = DatasetBundle.load(paths.DATASET_BUNDLE_DIR)
    aggr_route_lookup = pd.read_csv(paths.DATASETS_DIR + cfg.dataset.route_aggr + ".csv")
    print(db.train.x.shape)
    merged_data = db.train.x.merge(aggr_route_lookup, on="route_seq_hash")
    print(merged_data.shape)
    merged_data = merged_data[cfg.dataset.time_feature_names + cfg.dataset.route_feature_names]
    print(merged_data.shape)
    correlation_analysis(merged_data, db.train.y)

    mi = mutual_info_regression(merged_data, db.train.y)
    mi_series = pd.Series(mi, index=merged_data.columns).sort_values(ascending=False)
    print(mi_series)

    corr = merged_data.corrwith(db.train.y).sort_values(key=lambda x: abs(x), ascending=False)
    print(corr)

    feature_summary = pd.DataFrame({
        "MutualInfo": mi_series,
        "Correlation": corr
    }).sort_values("MutualInfo", ascending=False)
    feature_summary.to_parquet(f"{paths.RESULTS_DIR}/feature_selection/scores.parquet")




if __name__ == "__main__":
    main()
