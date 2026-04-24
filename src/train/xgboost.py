"""XGBoost baseline: training, optional grid search, and SHAP analysis."""
import itertools
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from config.config import Config
from data.dataset_bundle import DatasetBundle


def _sample_trips_per_route(
    x_df: pd.DataFrame,
    y_df: pd.Series,
    n_trips_per_route: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = x_df.copy()
    df["label"] = y_df
    df = df.groupby("stop_to_stop_id", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), n_trips_per_route), random_state=random_state)
    )
    return df.drop(columns=["label"]), df["label"]


def _merge_route_features(
    x_df: pd.DataFrame,
    route_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = x_df.merge(route_df, how="left", on="route_seq_hash", suffixes=("", "_route"))
    ids = merged[["id"]]
    merged = merged.drop(["id", "route_seq_hash", "stop_to_stop_id"], axis=1)
    return merged, ids


def xgboost_gridsearch(cfg: Config, db: DatasetBundle, route_df: pd.DataFrame) -> dict:
    """Random grid search over XGBoost hyperparameters using 3-fold CV."""
    X_sampled, y_train = _sample_trips_per_route(
        db.train.x, db.train.y, n_trips_per_route=10, random_state=cfg.training.random_state
    )
    X_train, _ = _merge_route_features(X_sampled, route_df)
    dtrain = xgb.DMatrix(X_train, label=y_train)

    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_lambda": [1, 5, 10],
        "reg_alpha": [0, 0.1, 1],
    }
    device = "cpu" if cfg.dataset.use_subset else "cuda"

    best_score = float("inf")
    best_params = None
    best_num_boost = None

    keys, values = zip(*param_grid.items())
    total_iterations = int(np.prod([len(v) for v in values]))
    for combo in tqdm(itertools.product(*values), total=total_iterations):
        params = dict(zip(keys, combo))
        params.update({
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": device,
            "eval_metric": "mae",
        })
        cv_results = xgb.cv(
            params=params,
            dtrain=dtrain,
            nfold=3,
            metrics="mae",
            num_boost_round=2000,
            early_stopping_rounds=50,
            seed=cfg.training.random_state,
            verbose_eval=False,
        )
        mean_mae = cv_results["test-mae-mean"].min()
        boost_rounds = int(cv_results["test-mae-mean"].argmin())
        if mean_mae < best_score:
            best_score = mean_mae
            best_params = params
            best_num_boost = boost_rounds

    print(f"Best params: {best_params}\nBest CV MAE: {best_score:.3f}\nBest n_estimators: {best_num_boost}")
    return {"params": best_params, "num_boost_round": best_num_boost, "cv_mae": best_score}


def _build_xgb_params(cfg: Config) -> dict:
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cpu" if cfg.dataset.use_subset else "cuda",
        "eval_metric": "mae",
        "max_depth": cfg.model.xgboost.max_depth,
        "learning_rate": cfg.model.xgboost.learning_rate,
        "subsample": cfg.model.xgboost.subsample,
        "colsample_bytree": cfg.model.xgboost.colsample_bytree,
        "reg_lambda": cfg.model.xgboost.reg_lambda,
        "reg_alpha": cfg.model.xgboost.reg_alpha,
    }


def _save_shap_plots(model: xgb.Booster, X_test: pd.DataFrame, output_dir: Path) -> None:
    """Compute SHAP values on a sample of the test set and save bar + beeswarm plots."""
    n_samples = min(100_000, len(X_test))
    sampled_idx = np.random.choice(len(X_test), size=n_samples, replace=False)
    X_test_sub = X_test.iloc[sampled_idx].copy()

    # Pretty-print feature names with category tags for the publication-quality plots.
    allowed_traffic = {"pedestrian", "agricultural", "bicycle", "bus", "car",
                       "moped", "motor_scooter", "motorcycle", "trailer", "truck"}
    road_categories = {"street_perc", "cityroad_perc", "regional_perc", "residential_perc",
                       "local_perc", "unpaved_perc", "public_transport_perc", "rest_area_perc",
                       "highway_perc", "motorway_perc"}

    def _pretty(name: str) -> str:
        suffix = ""
        if name in allowed_traffic:
            suffix = " (AT)"
        elif name in road_categories:
            suffix = " (RC)"
        pretty = (name + suffix)
        pretty = (pretty
                  .replace("on_road_", "")
                  .replace("avg", "mean")
                  .replace("length", "Segment length")
                  .replace("_perc", "")
                  .replace("_", " "))
        return pretty[0].upper() + pretty[1:]

    X_test_sub.columns = [_pretty(c) for c in X_test_sub.columns]

    explainer = shap.TreeExplainer(model, X_test_sub)
    shap_values = explainer(X_test_sub)

    for kind, plot_fn in [
        ("bar", lambda: shap.plots.bar(shap_values, max_display=50, show=False)),
        ("beeswarm", lambda: shap.plots.beeswarm(shap_values, max_display=50, show=False, color_bar=True)),
    ]:
        plot_fn()
        ax = plt.gca()
        for label in ax.get_yticklabels():
            label.set_color("black")
            label.set_fontname("DejaVu Sans")
            label.set_fontsize(12)
        plt.xlabel("Mean absolute SHAP value" if kind == "bar" else "SHAP value")
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_{kind}.pdf", bbox_inches="tight")
        plt.close()


def train_xgb(cfg: Config, db: DatasetBundle, route_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, _ = _merge_route_features(db.train.x, route_df)
    X_test, ids = _merge_route_features(db.test.x, route_df)

    dtrain = xgb.DMatrix(X_train, label=db.train.y, feature_names=X_train.columns.tolist())
    dtest = xgb.DMatrix(X_test, label=db.test.y, feature_names=X_test.columns.tolist())

    if cfg.dataset.use_validation:
        X_val, _ = _merge_route_features(db.val.x, route_df)
        dval = xgb.DMatrix(X_val, label=db.val.y, feature_names=X_val.columns.tolist())
        evals = [(dtrain, "train"), (dval, "val")]
        early_stopping_rounds = 50
        num_boost_round = 2000
    else:
        evals = [(dtrain, "train")]
        early_stopping_rounds = None
        num_boost_round = cfg.model.xgboost.num_boost_round

    model = xgb.train(
        params=_build_xgb_params(cfg),
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    if cfg.dataset.use_validation:
        print(f"XGBoost best iteration: {model.best_iteration} | best score: {model.best_score:.3f}")
    model.save_model(str(output_dir / "xgboost.json"))

    y_pred = model.predict(dtest)
    print(f"XGBoost test MAE: {mean_absolute_error(db.test.y, y_pred):.3f}")

    _save_shap_plots(model, X_test, output_dir)

    return pd.DataFrame({
        "id": ids["id"].values,
        "prediction": y_pred,
        "target": db.test.y,
    })
