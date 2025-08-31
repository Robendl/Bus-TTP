import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
import seaborn as sns


def plot_tac(margins, accuracies, metric, output_dir):
    for name, abs_accuracies in accuracies.items():
        plt.plot(margins, abs_accuracies, label=name)

    plt.xlabel(f'Tolerance margin ({'%' if metric == 'p' else metric})')
    plt.ylabel('Accuracy within margin')
    plt.title(f'{'Absolute' if metric == 's' else 'Relative'} Tolerance Accuracy Curve')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f'{output_dir}/tolerance_acc_curve_{metric}.png')
    plt.clf()
    plt.close()

def plot_error_histogram(errors = pd.Series, baseline=False):
    threshold = 200
    max_error = int(errors.max()) + 1
    errors_capped = np.copy(errors)
    errors_capped[errors_capped > threshold] = threshold + 3
    plt.hist(errors, bins=100,  edgecolor='black', alpha=0.7)
    plt.title(f'Error Histogram (200-{max_error} merged)')
    plt.xlabel('Prediction Error (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True)
    # plt.ylim(0, 32000)
    # plt.legend()
    plt.tight_layout()
    output_dir = HydraConfig.get().run.dir
    plt.savefig(f'{output_dir}/{'bs_' if baseline else ''}error_histogram.png')
    plt.clf()
    plt.close()

def plot_error_per_target_size(df: pd.DataFrame):
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
    output_dir = HydraConfig.get().run.dir
    plt.savefig(f'{output_dir}/error_target_size.png')
    plt.clf()
    plt.close()

def plot_losses(train_losses, val_losses, model_name):
    plt.plot(train_losses, label=f'Train')
    plt.plot(val_losses, label=f'Validation')
    # plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline {type} = {baseline:.2f}')
    plt.title(f'{model_name} Training Losses (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel(f'Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_dir = HydraConfig.get().run.dir
    plt.savefig(f'{output_dir}/{model_name}_losses.png')
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
    plt.grid(True)
    plt.tight_layout()
    output_dir = HydraConfig.get().run.dir
    plt.savefig(f'{output_dir}/losses.png')
    plt.clf()
    plt.close()

def calculate_zscores(df: pd.DataFrame):
    df["mean_elapsed_time"] = df.groupby("route_seq_hash")["recorded_elapsed_time"].transform("mean")
    df["std"] = df.groupby("route_seq_hash")["recorded_elapsed_time"].transform("std")
    df["zscore"] = (df["recorded_elapsed_time"] - df["mean_elapsed_time"]) / df["std"]
    return df["zscore"]

def plot_deviation(df: pd.DataFrame, df_filtered: pd.DataFrame, new_fraction, lower=0, upper=0, log_scale=False):
    s1 = calculate_zscores(df)
    s2 = calculate_zscores(df_filtered)

    if lower != 0 and upper != 0:
        s1 = s1.clip(lower, upper)
        s2 = s2.clip(lower, upper)

    removed_pct = 100 * (1 - new_fraction)
    plt.hist(s1, bins=100, alpha=0.5, label="Before", density=True)
    plt.hist(s2, bins=100, alpha=0.5, label=f"After (-{removed_pct:.1f}%)", density=True)
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
    plt.savefig(f'{output_dir}/{filename}.png')
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
    plt.grid(True)
    plt.tight_layout()
    plt.show()

