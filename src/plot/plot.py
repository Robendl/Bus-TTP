import matplotlib.pyplot as plt

def plot_scores(score_list, baseline, type):
    plt.figure(figsize=(8, 5))
    plt.plot(score_list, marker='o', label=f'{type} values')
    plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline {type} = {baseline}')
    plt.title(f'Model {type} Comparison')
    plt.xlabel('Model Index')
    plt.ylabel(f'{type}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_results(mae_list, mse_list, baseline_mae, baseline_mse):
    plot_scores(mae_list, baseline_mae, 'MAE')
    plot_scores(mse_list, baseline_mse, 'MSE')
