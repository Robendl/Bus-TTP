import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import sklearn.feature_selection as fs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

MI_THRESHOLD = 0.7
CORR_THRESHOLD = 0.6


def get_data():
    data_url = "https://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    labels = raw_df.values[1::2, 2]

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, labels


def train_loop(model, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Train the model
    epochs = 100
    for epoch in range(epochs):
        model.train_mlp()
        optimizer.zero_grad()
        predictions = model(X_train_tensor).squeeze()
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).squeeze()
        test_loss = criterion(test_predictions, y_test_tensor)
        score = r2_score(y_test_tensor, test_predictions)
        print(f"Test Loss: {test_loss.item()}, r2 score: {score}")


def mutual_info(X_train, X_test, y_train):
    # Feature selection using mutual information regression
    mi_scores = fs.mutual_info_regression(X_train, y_train)
    total_mi = sum(mi_scores)
    sorted_feature_idxs = np.flip(np.argsort(mi_scores))
    summed_mi = 0
    selected_features = []
    for feature_idx in sorted_feature_idxs:
        summed_mi += mi_scores[feature_idx]
        selected_features.append(feature_idx)
        if summed_mi > MI_THRESHOLD * total_mi:
            break

    print(f"Selected features: {len(selected_features)}")

    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    return X_train_selected, X_test_selected


def correlation_analysis(X_train, X_test):
    df = pd.DataFrame(X_train)
    corr_matrix = df.corr()

    plt.figure(figsize=(max(10, 0.5 * len(df.columns)), max(8, 0.5 * len(df.columns))))

    sns.set(font_scale=0.7)
    heatmap = sns.heatmap(
        corr_matrix,
        square=True,
        annot=True,
        fmt='.2f',
        linecolor='black',
        linewidths=0.5,
        cmap='coolwarm',
        cbar=True
    )

    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha="right")
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

    plt.title('Correlation Heatmap', fontsize=12)
    plt.tight_layout()  # voorkomt clipping
    plt.savefig("results/feature_selection/corr_mat.png", dpi=300)
    plt.clf()
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    # print(f"Dropping columns: {to_drop}")
    # selected_X_train = df.drop(df.columns[to_drop], axis=1).to_numpy()
    # selected_X_test = pd.DataFrame(X_test).drop(df.columns[to_drop], axis=1).to_numpy()
    # return selected_X_train, selected_X_test


def main():
    data, labels = get_data()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    X_train, X_test = correlation_analysis(X_train, X_test)
    # X_train, X_test = mutual_info(X_train, X_test, y_train)

    model = nn.Sequential(
        nn.Linear(len(X_train[0]), 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    train_loop(model, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)


if __name__ == "__main__":
    main()
