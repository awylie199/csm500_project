"""Rebalance Target Field"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


def plot_target_distribution(x_train: pd.DataFrame, y_train: pd.DataFrame, label: str) -> None:
    """Plot Target Distribution with Analysis"""
    print(f"X-Train Null Values:\n{x_train.isna().sum()}")
    print(f"Y-Train Null Values:\n{y_train.isna().sum()}")
    print(f"X-Train Size:\n{len(x_train)}")
    print(f"Y-Train Size:\n{len(y_train)}")
    assert len(x_train) == len(y_train)

    ax: Axes = y_train.value_counts().plot(kind="bar", figsize=(8, 8), title=label)
    _ = ax.set_xlabel("Application Result")
    _ = ax.set_ylabel("Frequency")
    plt.show()
