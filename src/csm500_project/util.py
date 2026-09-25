"""Project Utilities"""

from collections.abc import Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.preprocessing import label_binarize


def is_not_empty(value: object) -> bool:
    """Check if a value is Empty (i.e. not Falsy)"""

    return isinstance(value, str) and bool(value.strip())


def get_plot_lims(gdf: gpd.GeoDataFrame | gpd.GeoSeries) -> tuple[tuple[float, float], tuple[float, float]]:
    """Get Plot Limits with Padding"""

    minx, miny, maxx, maxy = gdf.total_bounds
    xrange: float = maxx - minx
    yrange: float = maxy - miny
    padding: float = 0.05  # 5%

    return (
        (minx - xrange * padding, maxx + xrange * padding),
        (miny - yrange * padding, maxy + yrange * padding),
    )


def is_within_area(x):
    return (x.notna().any()).astype(int)


def split_primary_physical_datasets(
    x_df: pd.DataFrame | gpd.GeoDataFrame, y_df: pd.DataFrame | gpd.GeoDataFrame, after_encoding: bool = True
) -> dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    encoded_physical_cols: list[str] = ["Property Type_0", "Property Type_1", "Property Type_2"] if after_encoding else ["Property Type"]
    physical_cols: list[str] = [*encoded_physical_cols, "Is Domestic", "Floor Area (m^2)"]
    not_na = x_df[physical_cols].notna().all(axis=1)

    return {
        "x_primary": x_df.copy(deep=True).drop(columns=physical_cols),
        "x_physical": x_df.copy(deep=True).loc[not_na],
        "y_primary": y_df.copy(deep=True),
        "y_physical": y_df.copy(deep=True).loc[not_na],
    }


def print_confusion_matrix(y_test, y_pred, title: str) -> None:
    """Print a confusion matrix for the model"""
    _, ax = plt.subplots(figsize=(8, 8))
    _ = ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred, ax=ax)
    ax.set_title(title)
    plt.show()


def print_calibration_display(classes: Sequence[tuple[int, str]], results, y_true, title: str) -> None:
    """Print a calibration display for the model"""
    _, ax = plt.subplots(figsize=(8, 8))

    for i, class_name in enumerate(classes):
        class_results = results[:, i]
        print(class_name, class_results)
        y_test_binary = (y_true == class_name).astype(int)

        _ = CalibrationDisplay.from_predictions(y_test_binary, class_results, n_bins=10, ax=ax, name=f"Class {class_name}")

    ax.set_title(title)
    plt.show()


def print_precision_recall_display(classes: Sequence[tuple[int, str]], results, y_true, title: str, is_probabilistic: bool) -> None:
    """Print a precision/recall display for the model"""
    _, ax = plt.subplots(figsize=(8, 8))

    for i, class_name in enumerate(classes):
        class_results = results[:, i]

        if is_probabilistic:
            print(class_name, class_results)
            y_test_binary = (y_true == class_name).astype(int)
        else:
            y_test_binary = label_binarize(y_true, classes=classes)[:, i]

        _ = PrecisionRecallDisplay.from_predictions(y_test_binary, class_results, ax=ax, name=f"Class {class_name}")

    ax.set_title(title)
    plt.show()


def print_roc_curve_display(classes: Sequence[tuple[int, str]], results, y_true, title: str) -> None:
    """Print a ROC curve display for the model"""
    _, ax = plt.subplots(figsize=(8, 8))

    for i, class_name in enumerate(classes):
        class_results = results[:, i]
        print(class_name, class_results)
        y_test_binary = (y_true == class_name).astype(int)

        _ = RocCurveDisplay.from_predictions(y_test_binary, class_results, ax=ax, name=f"Class {class_name}")

    ax.set_title(title)
    plt.show()
