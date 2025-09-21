import torch.multiprocessing as mp
from tqdm import tqdm

from feature_selection.correlation_analysis import correlation_analysis
from plot.analysis import validation_analysis, get_od_results, bootstrap_ci, paired_significance_test
from train.xgboost import xgboost_gridsearch, train_xgb

mp.set_start_method("spawn", force=True)
import pickle
import hydra
import torch
import numpy as np
from hydra.core.hydra_config import HydraConfig

from data.dataset_bundle import DatasetBundle
from data.mapping_dataset import seq_collate_fn, aggr_collate_fn
from plot.plot import plot_tac, scores_boxplot
from config.config import Config
from data.data_conversions import data_conversions, load_route_lookup
from data.data_processing import create_dataloaders
from model.lstm import LSTMFeedforwardCombination
from model.mlp import MLP
import config.paths as paths
from train.linear_regression import linear_regression
from train.train import train_model
from train.eval import evaluate

import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["HYDRA_FULL_ERROR"] = "1"

def run_training(cfg, model, route_lookup, dataset_bundle, num_workers, cfg_optim, device, output_dir, is_route_sequence):
    train_loader, val_loader, test_loader = create_dataloaders(cfg, dataset_bundle, route_lookup,
                                                               is_route_sequence, num_workers)
    train_losses, val_losses, val_id_targets, val_mae = train_model(cfg, model, train_loader, val_loader, cfg_optim, device)

    model_dir = f"{output_dir}/{model.name}"
    os.makedirs(model_dir, exist_ok=True)

    (mae, mape, rmse), abs_accuracies, relative_accuracies, test_id_targets, raw_scores = evaluate(cfg, model, test_loader, device)
    test_id_targets.to_parquet(f"{model_dir}/{cfg.dataset.time}_id_targets.parquet")

    results = test_id_targets.merge(dataset_bundle.test.x[["id", "stop_to_stop_id"]], on="id", how="left")
    od_results = get_od_results(results)
    bootstrap, result_string = bootstrap_ci(od_results, seed=cfg.training.random_state, model_name=model.name)
    print(result_string)
    print(f"{model.name} Test MAE: {mae:.3f}, MAPE: {mape:.3f}, RMSE: {rmse:.3f} ")
    validation_analysis(test_id_targets, model_dir, split="test", use_subset=cfg.dataset.use_subset)

    mae_path = os.path.join(output_dir, f"{model.name}_mae.txt")
    with open(mae_path, "w") as f:
        if cfg.dataset.use_validation:
            f.write(f"Val MAE: {val_mae:.3f}\n")
        f.write(f"Test MAE: {mae:.3f}, MAPE: {mape:.3f}, RMSE; {rmse:.3f} \n")

    if cfg.save_results:
        accuracies_dir = paths.RESULTS_DIR
        np.save(f"{accuracies_dir}/{model.name}_{cfg.dataset.time}_abs.npy", abs_accuracies)
        np.save(f"{accuracies_dir}/{model.name}_{cfg.dataset.time}_rel.npy", relative_accuracies)

    np.save(f"{model_dir}/{cfg.dataset.time}_abs.npy", abs_accuracies)
    np.save(f"{model_dir}/{cfg.dataset.time}_rel.npy", relative_accuracies)
    np.save(f"{model_dir}/{cfg.dataset.time}_train_losses.npy", train_losses)

    if cfg.dataset.use_validation:
        val_dir = model_dir + "_val"
        os.makedirs(val_dir, exist_ok=True)
        val_id_targets.to_parquet(f"{val_dir}/{cfg.dataset.time}_id_targets.parquet")
        print(f"{model.name} Val MAE: {val_mae:.3f}")
        validation_analysis(val_id_targets, val_dir, split="val", use_subset=cfg.dataset.use_subset)
        np.save(f"{model_dir}/{cfg.dataset.time}_val_losses.npy", val_losses)

    return abs_accuracies, relative_accuracies, raw_scores, od_results["MAE"], result_string

def load_results(cfg: Config, model_name):
    abs_accuracies = np.load(f"{paths.RESULTS_DIR}{model_name}_{cfg.dataset.time}_abs.npy")
    relative_accuracies = np.load(f"{paths.RESULTS_DIR}{model_name}_{cfg.dataset.time}_rel.npy")
    # id_targets = np.load(f"{baseline_dir}/id_targets.npy")
    return abs_accuracies, relative_accuracies

@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    print(f"Using dataset: {cfg.dataset.time}")
    if cfg.pre_data_conversions:
        data_conversions(cfg)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading time data")
    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR
                                        + ("_val" if cfg.dataset.use_validation else "")
                                        + ("_pca" if cfg.dataset.pca else ""),
                                        cfg.dataset.use_validation)
    print(dataset_bundle.train.x.shape)

    id_targets_dict = {}
    result_strings = []

    output_dir = HydraConfig.get().run.dir
    seq_route_lookup = None
    aggr_route_lookup = None
    num_workers = 4 if device.type == 'cuda' else 0
    print(f"num workers: {num_workers}")
    abs_accuracies_dict = {}
    relative_accuracies_dict = {}

    if cfg.compute_baseline or cfg.train_mlp or cfg.fit_xgboost:
        print("Loading aggregated route lookup", flush=True)
        aggr_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_aggr)

    baseline_dir = f"{paths.RESULTS_DIR}/baseline"
    if cfg.compute_baseline and not cfg.dataset.pca:
        print("Computing baseline", flush=True)
        lr_val_mae, (lr_mae, lr_mape, lr_rmse), abs_accuracies, relative_accuracies, id_targets = linear_regression(cfg, dataset_bundle, aggr_route_lookup)
        id_targets_dict["Linear Regression"] = id_targets
        np.save(f"{baseline_dir}/id_targets.npy", id_targets)
        results = id_targets.merge(dataset_bundle.test.x[["id", "stop_to_stop_id"]], on="id", how="left")
        od_results = get_od_results(results)
        bootstrap, result_string = bootstrap_ci(od_results, seed=cfg.training.random_state, model_name="Linear Regression")
        print(result_string)
        result_strings.append(result_string)
        print(f"Baseline MAE | val: {lr_val_mae:.3f}, test: MAE: {lr_mae:.3f}, MAPE: {lr_mape:.3f}, RMSE: {lr_rmse:.3f}", flush=True)
        os.makedirs(baseline_dir, exist_ok=True)
        abs_accuracies_dict["Linear Regression"] = abs_accuracies
        relative_accuracies_dict["Linear Regression"] = relative_accuracies
        np.save(f"{baseline_dir}/scores.npy", [lr_val_mae, lr_mae, lr_mape, lr_rmse])
        np.save(f"{baseline_dir}/abs_accuracies.npy", abs_accuracies)
        np.save(f"{baseline_dir}/rel_accuracies.npy", relative_accuracies)
    else:
        print("Loading baseline results", flush=True)
        scores = np.load(f"{baseline_dir}/scores.npy")
        lr_val_mae, lr_mae, lr_mape, lr_rmse = scores
        print(f"Baseline MAE | val: {lr_val_mae:.3f}, test: MAE: {lr_mae:.3f}, MAPE: {lr_mape:.3f}, RMSE: {lr_rmse:.3f}", flush=True)
        abs_accuracies = np.load(f"{baseline_dir}/abs_accuracies.npy")
        relative_accuracies = np.load(f"{baseline_dir}/rel_accuracies.npy")
        abs_accuracies_dict["Linear Regression"] = abs_accuracies
        relative_accuracies_dict["Linear Regression"] = relative_accuracies
        # id_targets_dict["Linear Regression"] = np.load(f"{baseline_dir}/id_targets.npy")

    if cfg.fit_xgboost:
        # xgboost_gridsearch(cfg, dataset_bundle, aggr_route_lookup)
        id_targets = train_xgb(cfg, dataset_bundle, aggr_route_lookup)
        results = id_targets.merge(dataset_bundle.test.x[["id", "stop_to_stop_id"]], on="id", how="left")
        od_results = get_od_results(results)
        bootstrap, result_string = bootstrap_ci(od_results, seed=cfg.training.random_state, model_name="XGBoost")
        print(result_string)
        result_strings.append(result_string)

    if cfg.train_mlp:
        input_dim = dataset_bundle.train.x.shape[1] - 3 + next(iter(aggr_route_lookup.values())).shape[1]
        model = MLP(cfg, input_dim)
        model.to(device)
        abs_accuracies, relative_accuracies, id_targets, mlp_od_mae, result_string = run_training(cfg, model, aggr_route_lookup,
                                                           dataset_bundle, num_workers, cfg.training.optimizer_mlp,
                                                           device, output_dir, is_route_sequence=False)
        result_strings.append(result_string)
        abs_accuracies_dict[model.name] = abs_accuracies
        relative_accuracies_dict[model.name] = relative_accuracies
        id_targets_dict["MLP"] = id_targets
    else:
        model_name = "MLP"
        abs_accuracies, relative_accuracies = load_results(cfg, model_name)
        abs_accuracies_dict[model_name] = abs_accuracies
        relative_accuracies_dict[model_name] = relative_accuracies

    if cfg.train_lstm:
        print("Loading sequence route lookup", flush=True)
        seq_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_seq)
        lstm_input_dim = next(iter(seq_route_lookup.values())).shape[1]
        ff_input_dim = dataset_bundle.train.x.shape[1] - 3
        model = LSTMFeedforwardCombination(cfg, lstm_input_dim, ff_input_dim)
        model.to(device)

        abs_accuracies, relative_accuracies, id_targets, lstm_od_mae, result_string = run_training(cfg, model, seq_route_lookup,
                                                           dataset_bundle, num_workers, cfg.training.optimizer_lstm,
                                                           device, output_dir, is_route_sequence=True)
        result_strings.append(result_string)
        id_targets_dict[model.name] = id_targets
        abs_accuracies_dict[model.name] = abs_accuracies
        relative_accuracies_dict[model.name] = relative_accuracies
    # else:
        # model_name = "LSTM"
        # abs_accuracies, relative_accuracies = load_results(cfg, model_name)
        # abs_accuracies_dict[model_name] = abs_accuracies
        # relative_accuracies_dict[model_name] = relative_accuracies

    with open(output_dir + "/final_scores.txt", "w") as f:
        for s in result_strings:
            f.write(s + "\n")

    if cfg.train_lstm and cfg.train_mlp:
        ptt_pvalue, w_pvalue = paired_significance_test(mlp_od_mae, lstm_od_mae)
        print(f"Paired t-test p = {ptt_pvalue:.5f}")
        print(f"Wilcoxon p = {w_pvalue:.5f}")
        path = os.path.join(output_dir, f"paired_significance.txt")
        with open(path, "w") as f:
            f.write(f"Paired t-test p = {ptt_pvalue:.5f} \n")
            f.write(f"Wilcoxon p = {w_pvalue:.5f} \n")
    # scores_boxplot(id_targets_dict)
    margins = np.arange(1, cfg.plot.margins_max, cfg.plot.step_size)
    plot_tac(margins, abs_accuracies_dict, 's', output_dir)
    margins = np.arange(1, cfg.plot.percentages_max, cfg.plot.step_size)
    plot_tac(margins, relative_accuracies_dict, 'p', output_dir)


if __name__ == "__main__":
    main()
