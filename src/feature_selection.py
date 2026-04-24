"""Feature selection diagnostics: Pearson correlation and mutual information."""
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm

from config import paths
from config.config import Config
from data.dataset_bundle import DatasetBundle
from runtime import setup_environment

setup_environment()


def correlation_matrix(X: pd.DataFrame, output_path: Path) -> None:
    corr_matrix = X.corr()
    n_features = len(X.columns)
    cell_size = 0.5
    fig_size = cell_size * n_features

    plt.figure(figsize=(fig_size, fig_size))
    sns.set(font_scale=0.7)

    heatmap = sns.heatmap(
        corr_matrix,
        square=True, annot=True, fmt=".2f",
        linecolor="black", linewidths=0.5,
        cmap="coolwarm", cbar=True, cbar_kws={"shrink": 0.8},
    )
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha="right")
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

    plt.title("Correlation Heatmap", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _combine_cyclical(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Decode (sin_X, cos_X) cyclical encoding into a single [0, 1) column X."""
    df[name] = np.arctan2(df[f"sin_{name}"], df[f"cos_{name}"])
    df[name] = (df[name] + np.pi) / (2 * np.pi)
    return df.drop([f"sin_{name}", f"cos_{name}"], axis=1)


def combine_scores(corr: pd.DataFrame, mi: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize correlation and MI into a single ranking score."""
    corr = corr.reset_index()
    mi = mi.reset_index()
    corr.columns = ["feature", "score_corr"]
    mi.columns = ["feature", "score_mi"]
    merged = pd.merge(corr, mi, on="feature")

    for col, scaled_col in (("score_corr", "corr_scaled"), ("score_mi", "mi_scaled")):
        rng = merged[col].max() - merged[col].min()
        merged[scaled_col] = (merged[col] - merged[col].min()) / rng

    merged["combined_score"] = merged["corr_scaled"] + merged["mi_scaled"]
    ranked = merged.sort_values("combined_score", ascending=False)
    return ranked[["feature", "score_corr", "score_mi", "combined_score"]]


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    output_dir = Path(paths.RESULTS_DIR) / "feature_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    db = DatasetBundle.load(paths.DATASET_BUNDLE_DIR, cfg)
    aggr_route_lookup = pd.read_parquet(paths.DATASETS_DIR + cfg.dataset.route_aggr + ".parquet")

    merged = db.train.x.merge(aggr_route_lookup, on="route_seq_hash")
    merged = merged[cfg.dataset.time_feature_names + cfg.dataset.route_feature_names].copy()
    for cyclical in ("time", "day", "year"):
        merged = _combine_cyclical(merged, cyclical)

    correlation_matrix(merged, output_dir / "corr_mat.png")

    corr = merged.corrwith(db.train.y).sort_values(key=lambda x: abs(x), ascending=False)
    corr.to_frame().to_parquet(output_dir / "corr.parquet")

    sample = merged.sample(n=int(0.2 * merged.shape[0]), random_state=cfg.training.random_state)
    y_sample = db.train.y.loc[sample.index]

    def mi_score(feature: str) -> float:
        return float(mutual_info_regression(sample[[feature]], y_sample)[0])

    scores = Parallel(n_jobs=-1)(delayed(mi_score)(f) for f in tqdm(sample.columns.tolist()))
    mi = pd.Series(scores, index=sample.columns).sort_values(ascending=False)
    mi.to_frame().to_parquet(output_dir / "mi.parquet")

    ranking = combine_scores(corr, mi)
    ranking.to_parquet(output_dir / "ranking.parquet")
    print(ranking)


if __name__ == "__main__":
    main()
