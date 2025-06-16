import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig

def plot_scores(score_list, baseline, type):
    plt.figure(figsize=(8, 5))
    plt.plot(score_list, marker='o', label=f'{type} values')
    plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline {type} = {baseline}:.2f')
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
