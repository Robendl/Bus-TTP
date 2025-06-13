import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def cyclical_to_hour(sin_val, cos_val):
    angle = np.arctan2(sin_val, cos_val)  # returns value in [-π, π]
    normalized = (angle + np.pi) / (2 * np.pi)  # normalize to [0, 1]
    hour = normalized * 24  # scale to [0, 24]
    return hour

def cyclical_to_unit(sin_val, cos_val):
    angle = np.arctan2(sin_val, cos_val)
    return (angle + np.pi) / (2 * np.pi)

def plot_distribution(X, y):
    df = X.copy()
    df['recorded_elapsed_time'] = y.squeeze()

    # Step 2: Compute group-wise mean and delta
    df['group_mean_time'] = df.groupby('stop_to_stop_id')['recorded_elapsed_time'].transform('mean')
    df['delta_elapsed_time'] = df['recorded_elapsed_time'] - df['group_mean_time']

    df['hour_of_day'] = cyclical_to_unit(df['sin_time'], df['cos_time']) * 24
    df['day_of_week'] = cyclical_to_unit(df['sin_day'], df['cos_day']) * 7

    # Step 4: Plot 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Loop over each group to draw connected lines
    for _, group in df.groupby('stop_to_stop_id'):
        group = group.sort_values(by=['day_of_week', 'hour_of_day'])  # optional sort
        ax.plot(
            group['hour_of_day'],
            group['day_of_week'],
            group['delta_elapsed_time'],
            alpha=0.6
        )

    ax.set_xlabel('Hour of Day')
    ax.set_zlabel('Δ Elapsed Time')
    ax.set_ylabel('Day of Week')
    ax.set_title('Δ Elapsed Time vs Hour of Day and Day of Week')
    plt.tight_layout()
    plt.show()


def plot_distribution1(X, y):
    df = X.copy()
    df['recorded_elapsed_time'] = y.squeeze()

    df['group_mean_time'] = df.groupby('stop_to_stop_id')['recorded_elapsed_time'].transform('mean')

    df['delta_elapsed_time'] = df['recorded_elapsed_time'] - df['group_mean_time']

    df['hour_of_day'] = cyclical_to_hour(df['sin_time'], df['cos_time'])

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='hour_of_day',
        y='delta_elapsed_time',
        alpha=0.5
    )
    plt.title('Deviation from Mean Elapsed Time by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Δ Elapsed Time (seconds or minutes)')
    plt.xticks(range(0, 25, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()