import hydra
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_regression

from config import paths
from config.config import Config
from data.data_conversions import load_route_lookup
from data.dataset_bundle import DatasetBundle
from data.plot_distribution import plot_distribution

def correlation_matrix(X_train, X_test):
    df = pd.DataFrame(X_train)
    corr_matrix = df.corr()

    n_features = len(df.columns)
    cell_size = 0.5
    fig_width = cell_size * n_features
    fig_height = cell_size * n_features

    plt.figure(figsize=(fig_width, fig_height))

    sns.set(font_scale=0.7)
    heatmap = sns.heatmap(
        corr_matrix,
        square=True,
        annot=True,
        fmt='.2f',
        linecolor='black',
        linewidths=0.5,
        cmap='coolwarm',
        cbar=True,
        cbar_kws={"shrink": 0.8}
    )

    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha="right")
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

    plt.title('Correlation Heatmap', fontsize=12)
    plt.tight_layout()
    plt.savefig("results/feature_selection/corr_mat.png", dpi=300)
    plt.clf()

@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    db = DatasetBundle.load(paths.DATASET_BUNDLE_DIR)
    aggr_route_lookup = pd.read_parquet(paths.DATASETS_DIR + cfg.dataset.route_aggr + ".parquet")
    print(db.train.x.shape, flush=True)
    merged_data = db.train.x.merge(aggr_route_lookup, on="route_seq_hash")
    print(merged_data.shape, flush=True)
    merged_data = merged_data[cfg.dataset.time_feature_names + cfg.dataset.route_feature_names]
    print(merged_data.shape, flush=True)
    # correlation_matrix(merged_data, db.train.y)
    #
    # corr = merged_data.corrwith(db.train.y).sort_values(key=lambda x: abs(x), ascending=False)
    # corr.to_frame().to_parquet(f"{paths.RESULTS_DIR}/feature_selection/corr.parquet")
    # print(corr, flush=True)
    X_sample = merged_data.sample(n=int(0.2*merged_data.shape[0]), random_state=42)
    y_sample = db.train.y.loc[X_sample.index]

    def mi_score(feature):
        return mutual_info_regression(X_sample[[feature]], y_sample)[0]

    features = X_sample.columns.tolist()

    scores = Parallel(n_jobs=-1)(
        delayed(mi_score)(feature) for feature in features
    )

    mi_series = pd.Series(scores, index=X_sample.columns).sort_values(ascending=False)
    mi_series.to_frame().to_parquet(f"{paths.RESULTS_DIR}/feature_selection/mi.parquet")
    print(mi_series, flush=True)



if __name__ == "__main__":
    main()
