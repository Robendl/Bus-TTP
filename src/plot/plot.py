import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig

def plot_tac(margins, accuracies, metric, output_dir):
    for name, abs_accuracies in accuracies.items():
        plt.plot(margins, abs_accuracies, label=name)

    plt.xlabel(f'Tolerance margin (({'%' if metric == 'p' else metric}))')
    plt.ylabel('Accuracy within margin')
    plt.title(f'{'Absolute' if metric == 's' else 'Relative'} Tolerance Accuracy Curve)')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f'{output_dir}/tolerance_acc_curve_{metric}.png')
    plt.clf()
    plt.close()

def plot_error_histogram(errors, baseline=False):
    plt.hist(errors, bins=100,  edgecolor='black', alpha=0.7)
    plt.title('Error Histogram')
    plt.xlabel('Prediction Error (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True)
    # plt.ylim(0, 32000)
    plt.axvline(x=0, color='red', linestyle='--', label='Perfect prediction')
    plt.legend()
    plt.tight_layout()
    output_dir = HydraConfig.get().run.dir
    plt.savefig(f'{output_dir}/{'bs_' if baseline else ''}error_histogram.png')
    plt.clf()
    plt.close()

def plot_error_per_target_size(df):
    df_binned = df[df['target'] < 2000]

    bins = list(range(0, 2001, 200))  # [0, 200, 400, ..., 2000]
    labels = [f"{bins[i]}–{bins[i + 1]}" for i in range(len(bins) - 1)]
    df_binned["target_bin"] = pd.cut(df_binned["target"], bins=bins, labels=labels, right=False)

    grouped = df_binned.groupby('target_bin')['abs_error']
    bin_stats = grouped.agg(['mean', 'std']).reset_index()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(bin_stats['target_bin'], bin_stats['mean'], marker='o', label='Mean % Error')

    # Shaded area for ±1 std
    plt.fill_between(
        bin_stats['target_bin'].astype(str),
        bin_stats['mean'] - bin_stats['std'],
        bin_stats['mean'] + bin_stats['std'],
        alpha=0.3,
        label='±1 Std Dev'
    )

    plt.xticks(rotation=45)
    plt.xlabel("Target Bin")
    plt.ylabel("Mean Percentage Error")
    plt.title("Mean Percentage Error by Target Size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save figure
    output_dir = HydraConfig.get().run.dir
    plt.savefig(f'{output_dir}/error_target_size.png')

def plot_losses(train_losses, val_losses, model_name):
    plt.plot(train_losses, marker='o', label=f'Train')
    plt.plot(val_losses, marker='o', label=f'Validation')
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
