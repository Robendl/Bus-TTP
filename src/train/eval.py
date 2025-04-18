import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
import wandb

def evaluate(model, dataloader):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X
            batch_y = batch_y

            outputs = model(batch_X).squeeze()

            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())

    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)

    # Log to Weights & Biases
    wandb.log({"eval/mse": mse, "eval/mae": mae})

    print(f"\n📊 Eval Results — MSE: {mse:.4f}, MAE: {mae:.4f}\n")
    return mse, mae
