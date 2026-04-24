"""Diagnostic plots of the time/day-of-week distribution of trip duration."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cyclical_to_unit(sin_val: pd.Series, cos_val: pd.Series) -> pd.Series:
    """Decode (sin, cos) cyclical encoding back into [0, 1]."""
    angle = np.arctan2(sin_val, cos_val)
    return (angle + np.pi) / (2 * np.pi)


def plot_distribution(X: pd.DataFrame, y: pd.Series) -> None:
    """3D scatter of per-OD residual elapsed time vs hour of day vs day of week."""
    df = X.copy()
    df["recorded_elapsed_time"] = y.squeeze()
    df["group_mean_time"] = df.groupby("stop_to_stop_id")["recorded_elapsed_time"].transform("mean")
    df["delta_elapsed_time"] = df["recorded_elapsed_time"] - df["group_mean_time"]
    df["hour_of_day"] = cyclical_to_unit(df["sin_time"], df["cos_time"]) * 24
    df["day_of_week"] = cyclical_to_unit(df["sin_day"], df["cos_day"]) * 7

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    for _, group in df.groupby("stop_to_stop_id"):
        group = group.sort_values(by=["day_of_week", "hour_of_day"])
        ax.plot(group["hour_of_day"], group["day_of_week"], group["delta_elapsed_time"], alpha=0.6)

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.set_zlabel("Δ Elapsed Time")
    ax.set_title("Δ Elapsed Time vs Hour of Day and Day of Week")
    plt.tight_layout()
    plt.show()
