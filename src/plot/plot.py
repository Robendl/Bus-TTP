from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
import seaborn as sns
from matplotlib import ticker
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from tqdm import tqdm


def plot_tac(margins, accuracies, metric, output_dir):
    colors = plt.get_cmap("Set1")

    for idx, (name, abs_accuracies) in enumerate(accuracies.items()):
        plt.plot(margins, abs_accuracies, label=name, color=colors(idx))

    plt.xlabel(f'Tolerance margin ({'%' if metric == 'p' else metric})')
    plt.ylabel('Accuracy within margin')
    # plt.title(f'{'Absolute' if metric == 's' else 'Relative'} Tolerance Accuracy Curve')
    plt.ylim(0, 1)
    plt.legend(frameon=True, loc="lower right")
    plt.grid(alpha=0.3, linestyle='--')
    plt.savefig(f'{output_dir}/tolerance_acc_curve_{metric}.pdf')
    plt.clf()
    plt.close()

def tolerance_accuracy(targets, predictions, tolerance):
    targets = np.array(targets)
    predictions = np.array(predictions)
    errors = np.abs(targets - predictions)
    return np.mean(errors <= tolerance)

def bootstrap_tac_per_model(
    df_dict: Dict[str, pd.DataFrame],
    margins: np.ndarray,
    seed,
    output_dir,
    ci=95,
    n_boot=1000,
    percentage=False,
):
    rng = np.random.default_rng(seed)
    results = {}

    for model_name, df in df_dict.items():
        # --- 1. errors vooraf berekenen
        if percentage:
            errors = np.abs(df["prediction"].values - df["target"].values) / df["target"].values
            margins_arr = np.array(margins, dtype=float)
            margins_arr = margins_arr / 100.0
        else:
            errors = np.abs(df["prediction"].values - df["target"].values)
            margins_arr = np.array(margins, dtype=float)

        df_err = pd.DataFrame({
            "stop_to_stop_id": df["stop_to_stop_id"].values,
            "error": errors
        })

        # --- 2. accuracies per OD-pair en tolerance precomputen
        pair_acc = (
            df_err.groupby("stop_to_stop_id")["error"]
            .apply(lambda e: (e.values[:, None] <= margins_arr).mean(axis=0))
        )
        pair_acc = np.stack(pair_acc.values)  # shape: (n_pairs, n_margins)

        n_pairs = pair_acc.shape[0]
        tac_samples = np.empty((n_boot, len(margins_arr)))

        # --- 3. bootstrap alleen met indices
        for b in range(n_boot):
            idx = rng.integers(0, n_pairs, n_pairs)
            tac_samples[b] = pair_acc[idx].mean(axis=0)

        mean_curve = tac_samples.mean(axis=0)
        lower_curve = np.percentile(tac_samples, (100 - ci) / 2, axis=0)
        upper_curve = np.percentile(tac_samples, 100 - (100 - ci) / 2, axis=0)

        results[model_name] = {
            "mean": mean_curve,
            "lower": lower_curve,
            "upper": upper_curve,
        }

    # --- Plot
    colors = plt.get_cmap("Set1")
    for idx, (name, res) in enumerate(results.items()):
        color = colors(idx)
        plt.plot(margins, res["mean"], label=name, color=color)
        plt.fill_between(margins, res["lower"], res["upper"], color=color, alpha=0.2)
        relevant_margins = [10, 20, 30]
        for idx in range(len(margins)):
            if margins[idx] in relevant_margins:
                print(f"{name}, Margin {margins[idx]}, accuracy {res['mean'][idx]*100:.2f}")

    if percentage:
        plt.xlabel("Tolerance margin (%)")
    else:
        plt.xlabel("Tolerance margin (s)")

    plt.ylabel("Accuracy within margin")
    plt.ylim(0, 1)
    plt.legend(frameon=True, loc="lower right")
    plt.grid(alpha=0.3, linestyle="--")

    suffix = "_tac_percentage" if percentage else "_tac_absolute"
    plt.savefig(output_dir + f"/bootstrap{suffix}.pdf")
    plt.close()



def plot_error_histogram(errors: pd.Series, model_dir, baseline=False):
    colors = plt.get_cmap("Set1")
    threshold = 300
    max_error = int(errors.max()) + 1
    errors_capped = np.copy(errors)
    errors_capped[errors_capped > threshold] = threshold + 3
    plt.hist(errors_capped, bins=100,  edgecolor='black', alpha=0.7, color=colors(1), density=True)
    # plt.title(f'Error Histogram ({threshold}-{max_error} merged)')
    plt.xlabel('Percentage error')
    plt.ylabel('Density')
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    plt.grid(alpha=0.3, linestyle='--')
    plt.ylim(0, 0.022)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f'{model_dir}/{'bs_' if baseline else ''}error_histogram_({max_error}).pdf')
    plt.clf()
    plt.close()

def plot_error_per_target_size(df: pd.DataFrame, model_dir):
    max_target = int(df['target'].max())
    bins = list(range(0, 2001, 200))
    if max_target > 2000:
        bins = bins + [max_target]

    labels = [f"{bins[i]}–{bins[i + 1]}" for i in range(len(bins) - 1)]
    df["target_bin"] = pd.cut(df["target"], bins=bins, labels=labels, right=False)

    grouped = df.groupby('target_bin', observed=False)['abs_error']
    bin_stats = grouped.agg(['mean', 'std']).reset_index()

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(bin_stats['target_bin'], bin_stats['mean'], marker='o', label='Mean % Error')
    # Shaded area for ±1 std
    ax1.fill_between(
        bin_stats['target_bin'].astype(str),
        bin_stats['mean'] - bin_stats['std'],
        bin_stats['mean'] + bin_stats['std'],
        alpha=0.3,
        label='±1 Std Dev'
    )
    ax1.set_ylabel("Mean Percentage Error")

    fractions = df['target_bin'].value_counts().reindex(labels, fill_value=0) / df.shape[0]
    ax2 = ax1.twinx()
    ax2.plot(labels, fractions, color='red', marker='o')
    ax2.set_ylabel("Fraction of data")
    ax2.set_ylim(0, 1)

    plt.xticks(rotation=45)
    plt.xlabel("Target Bin")
    plt.title("Mean Percentage Error by Target Size")
    # plt.grid(True)
    # plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(f'{model_dir}/error_target_size.pdf')
    plt.clf()
    plt.close()

def plot_losses(train_losses, val_losses, model_name, output_dir=None):
    # sns.set_theme(style="whitegrid", palette="deep")
    colors = plt.get_cmap("Set1")

    plt.plot(range(1, len(train_losses) + 1),
             train_losses,
             label="Train",
             color=colors(1),
             linewidth=2)

    first_val_epoch = len(train_losses) - len(val_losses)

    if len(val_losses) > 0:
        plt.plot(range(first_val_epoch + 1, len(train_losses) + 1),
                 val_losses,
                 label="Validation",
                 color=colors(0),
                 linewidth=2)
    # plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline {type} = {baseline:.2f}')
    # plt.title(f'{model_name} Training Losses (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel(f'Huber loss')
    plt.ylim(top=40, bottom=10)
    plt.legend(frameon=True, loc="upper right")
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    if output_dir is None:
        output_dir = HydraConfig.get().run.dir
    plt.savefig(f'{output_dir}/{model_name}_losses.pdf')
    plt.clf()
    plt.close()

def plot_multiple_losses(train_losses_ls, val_losses_ls):
    for train_losses, val_losses in zip(train_losses_ls, val_losses_ls):
        plt.plot(train_losses, marker='o', label=f'Train', color='blue')
        plt.plot(val_losses, marker='o', label=f'Validation', color='red')
    # plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline {type} = {baseline:.2f}')
    plt.title(f'Training Losses (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel(f'Loss')
    plt.legend()
    plt.grid(alpha=0.3, linestyle='--')
    # plt.tight_layout()
    # output_dir = HydraConfig.get().run.dir
    # plt.savefig(f'{output_dir}/losses.png')
    plt.show()
    plt.clf()
    plt.close()

def calculate_zscores(df: pd.DataFrame):
    df["mean_elapsed_time"] = df.groupby("route_seq_hash")["recorded_elapsed_time"].transform("mean")
    df["std"] = df.groupby("route_seq_hash")["recorded_elapsed_time"].transform("std")
    df["zscore"] = (df["recorded_elapsed_time"] - df["mean_elapsed_time"]) / df["std"]
    return df["zscore"]

def plot_deviation(df: pd.DataFrame, df_filtered: pd.DataFrame, new_fraction, lower=0, upper=0, log_scale=False):
    colors = plt.get_cmap("Set1")
    s1 = calculate_zscores(df)
    s2 = calculate_zscores(df_filtered)

    if lower != 0 and upper != 0:
        s1 = s1.clip(lower, upper)
        s2 = s2.clip(lower, upper)

    removed_pct = 100 * (1 - new_fraction)
    plt.hist(s1, bins=100, alpha=0.5, label="Before", density=True, color=colors(0))
    plt.hist(s2, bins=100, alpha=0.5, label=f"After (-{removed_pct:.1f}%)", density=True, color=colors(1))
    filename = "z-scores"
    if log_scale:
        plt.yscale('log')
        plt.ylabel("log(Density)")
        filename += "_log"
    else:
        plt.ylabel("Density")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Z-score")
    plt.legend()
    output_dir = HydraConfig.get().run.dir
    plt.savefig(f'{output_dir}/{filename}.pdf')
    plt.clf()
    plt.close()


def plot_seq_length_distribution(df_route):
    sequence_lengths = df_route.groupby("route_seq_hash").size()
    print(sum(sequence_lengths > 90))

    # Plot de distributie
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=50)
    plt.title("Distributie van sequence lengtes per route_seq_hash")
    plt.xlabel("Sequence lengte (aantal wegvakken)")
    plt.ylabel("Aantal routes")
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

def scores_boxplot(id_targets_dict, output_dir=None):
    colors = plt.get_cmap("Set1")
    metrics = {
        "MAE": mean_absolute_error,
        "MAPE": mean_absolute_percentage_error,
        "RMSE": root_mean_squared_error
    }
    n_metrics = len(metrics)
    # mean_absolute_error(a, b, multioutput='raw_values')

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=False)

    for i, (metric, function) in enumerate(metrics.items()):
        data = []
        labels = []
        for model, id_targets in id_targets_dict.items():
            values = mean_absolute_error(np.array(id_targets["target"]), np.array(id_targets["prediction"]), multioutput='raw_values')
            print(values)
            data.append(values)
            labels.append(model)
        axes[i].boxplot(data, tick_labels=labels)
        axes[i].set_title(metric)
        axes[i].set_ylabel("Score")

    plt.tight_layout()
    if output_dir is None:
        output_dir = HydraConfig.get().run.dir
    plt.savefig(f'{output_dir}/scores_boxplot.pdf')
    plt.clf()
    plt.close()

def check_early_stopping(val_losses, min_delta=0.15, patience=3):
    best_loss = float("inf")
    wait = 0

    for epoch, loss in enumerate(val_losses):
        if loss < best_loss - min_delta:
            best_loss = loss
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            return True, epoch, best_loss  # dit is het epoch waarop early stopping zou triggeren

    return False, 0, 0


if __name__ == "__main__":
    train_list = []
    val_list = []
    dir = "results/gridsearches/losses_mlp/"
    under10 = 0
    cnt = 0
    best_loss = float("inf")
    best_i = 0
    tbd = []
    results = []
    for i in range(48, 49):
        train_losses = np.load(dir + f"train_{i}.npy")
        # print(len(train_losses))

        val_losses = np.load(dir + f"val_{i}.npy")
        stopped, epoch, loss = check_early_stopping(val_losses)
        print(val_losses)
        if stopped and loss < best_loss:
            best_i = i
            best_loss = loss

        train_list.append(train_losses)
        val_list.append(val_losses)

        if not stopped:
            tbd.append(i)
        else:
            results.append((i, loss))

    for r in results:
        print(f"{r[0]}, {r[1]:.2f}")
    print(best_i, best_loss)
    print(tbd)
    plot_multiple_losses(train_list, val_list)

    # dir = "results/pca_run/"
    # mlp_loss = np.load(dir + "MLP/dataset_time_val_losses.npy")
    # lstm_loss = np.load(dir + "LSTM/dataset_time_val_losses.npy")
    # print(min(mlp_loss), min(lstm_loss))

    #
    # id_targets_dict = {}
    # id_targets_dict["MLP"] = pd.read_parquet(f"{dir}/MLP/dataset_time_id_targets.parquet")
    # id_targets_dict["LSTM"] = pd.read_parquet(f"{dir}/LSTM/dataset_time_id_targets.parquet")
    # # scores_boxplot(id_targets_dict, output_dir=dir)
    # for model, id_targets in id_targets_dict.items():
    #     model_dir = f"{dir}/{model}/"
    #     id_targets["error"] = ((id_targets["prediction"] - id_targets["target"]) / id_targets["target"]) * 100
    #     # id_targets["abs_error"] = id_targets["error"].abs()
    #     # plot_error_per_target_size(id_targets.copy(), model_dir)
    #     plot_error_histogram(id_targets["error"].copy(), model_dir)


    # mlp_train = np.load(f"{dir}/MLP/dataset_time_train_losses.npy")
    # mlp_val = np.load(f"{dir}/MLP/dataset_time_val_losses.npy")
    # lstm_train = np.load(f"{dir}/LSTM/dataset_time_train_losses.npy")
    # lstm_val = np.load(f"{dir}/LSTM/dataset_time_val_losses.npy")
    #
    # plot_losses(mlp_train, mlp_val, "MLP", output_dir=dir)
    # plot_losses(lstm_train, lstm_val, "LSTM", output_dir=dir)
    # abs_margins = np.arange(1, 61, 5)
    # rel_margins = np.arange(1, 101, 5)
    # abs = {}
    # abs["MLP"] = np.load(f"{dir}/MLP/dataset_time_abs.npy")
    # abs["LSTM"] = np.load(f"{dir}/LSTM/dataset_time_abs.npy")
    # abs["Linear Regression"] = np.load(f"{dir}/baseline/abs_accuracies.npy")
    # rel = {}
    # rel["MLP"] = np.load(f"{dir}/MLP/dataset_time_rel.npy")
    # rel["LSTM"] = np.load(f"{dir}/LSTM/dataset_time_rel.npy")
    # rel["Linear Regression"] = np.load(f"{dir}/baseline/rel_accuracies.npy")
    # plot_tac(abs_margins, abs, 's', dir)
    # plot_tac(rel_margins, rel, 'p', dir)
