"""Rebalance Target Field"""

from typing import Final

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTEN, SMOTENC

from .discrete_data_analysis import FINAL_CATEGORIES
from .util import split_primary_physical_datasets

_SMOTE_SEED: Final[int] = 20260906


def _plot_target_distribution(target: pd.DataFrame, label: str) -> None:
    ax = target.value_counts().plot(kind="bar", figsize=(8, 8), title=label)
    ax.set_xlabel("Application Result")
    ax.set_ylabel("Frequency")
    plt.show()


def resample_target(x_train: pd.DataFrame | gpd.GeoDataFrame, y_train: pd.DataFrame | gpd.GeoDataFrame) -> dict[str, pd.DataFrame]:
    """Split the dataset into a Primary Dataset, and a smaller second dataset which contains the primary dataset and physical property columns with complete values."""

    # Balance the target field
    datasets = split_primary_physical_datasets(x_train, y_train)

    _plot_target_distribution(datasets["y_primary"], "Primary Target (Application Result) Distribution")
    _plot_target_distribution(datasets["y_physical"], "Physical Target (Application Result) Distribution")

    print(f"X-Train Primary Null Values:\n{datasets['x_primary'].isna().sum()}")
    print(f"Y-Train Primary Null Values:\n{datasets['y_primary'].isna().sum()}")
    print(f"X-Train Primary Size:\n{len(datasets['x_primary'])}")
    print(f"Y-Train Primary Size:\n{len(datasets['y_primary'])}")
    assert len(datasets["x_primary"]) == len(datasets["y_primary"])

    print(f"X-Train Physical Null Values:\n{datasets['x_physical'].isna().sum()}")
    print(f"Y-Train Physical Null Values:\n{datasets['y_physical'].isna().sum()}")
    print(f"Y-Train Physical Size:\n{len(datasets['x_physical'])}")
    print(f"Y-Train Physical Size:\n{len(datasets['y_physical'])}")
    assert len(datasets["x_physical"]) == len(datasets["y_physical"])

    smc: SMOTENC = SMOTENC(random_state=_SMOTE_SEED, categorical_features=FINAL_CATEGORIES)
    sm: SMOTEN = SMOTEN(random_state=_SMOTE_SEED)

    x_primary_resampled: pd.DataFrame
    y_primary_resampled: pd.DataFrame
    x_physical_resampled: pd.DataFrame
    y_physical_resampled: pd.DataFrame

    x_primary_resampled, y_primary_resampled = sm.fit_resample(datasets["x_primary"], datasets["y_primary"])  # pyright: ignore[reportAssignmentType]
    x_physical_resampled, y_physical_resampled = smc.fit_resample(datasets["x_physical"], datasets["y_physical"])  # pyright: ignore[reportAssignmentType]

    _plot_target_distribution(y_primary_resampled, "Primary Target (Application Result) SMOTEN Distribution")
    _plot_target_distribution(y_physical_resampled, "Physical Target (Application Result) SMOTENC Distribution")

    return {
        "x_primary_resampled": x_primary_resampled,
        "x_physical_resampled": x_physical_resampled,
        "y_primary_resampled": y_primary_resampled,
        "y_physical_resampled": y_physical_resampled,
    }
