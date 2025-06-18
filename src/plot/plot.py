import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig

def plot_tac(margins, accuracies, base_accuracies, metric, datasplit):
    plt.figure(figsize=(10, 5))
    plt.plot(margins, accuracies, label="MLP")
    plt.plot(margins, base_accuracies, label="Base")
    plt.xlabel(f'Tolerance margin ({'%' if metric == 'p' else metric})')
    plt.ylabel('Accuracy within margin')
    plt.title(f'Tolerance Accuracy Curve ({datasplit})')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.legend()
    output_dir = HydraConfig.get().run.dir
    plt.savefig(f'{output_dir}/{datasplit}_tolerance_acc_curve_{metric}.png')
    plt.clf()
    plt.close()

def plot_error_histogram(errors):
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=100, range=(-100, 100),  edgecolor='black', alpha=0.7)
    plt.title('Error Histogram')
    plt.xlabel('Prediction Error (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.axvline(x=0, color='red', linestyle='--', label='Perfect prediction')
    plt.legend()
    plt.tight_layout()
    output_dir = HydraConfig.get().run.dir
    plt.savefig(f'{output_dir}/error_histogram.png')
    plt.clf()
    plt.close()

def plot_scores(score_list, baseline, type):
    plt.figure(figsize=(8, 5))
    plt.plot(score_list, marker='o', label=f'{type} values')
    if type == 'MSE':
        plt.yscale('log')
    plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline {type} = {baseline:.2f}')
    plt.title(f'Model {type} Score')
    plt.xlabel('Epoch')
    plt.ylabel(f'{type}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_dir = HydraConfig.get().run.dir
    plt.savefig(f'{output_dir}/{type}.png')
    plt.clf()
    plt.close()

def plot_results(mae_list, mse_list, baseline_mae, baseline_mse):
    plot_scores(mae_list, baseline_mae, 'MAE')

    plot_scores(mse_list, baseline_mse, 'MSE')
