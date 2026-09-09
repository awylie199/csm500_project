"""Continuous Data Analysis"""

from typing import Final

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import boxcox, jarque_bera, kurtosis, median_abs_deviation, norm, normaltest, probplot, skew, trim_mean

_CONTINUOUS_DATA_FIELDS: Final[tuple[str, ...]] = ("Floor Area (m^2)",)


def _print_pass_normal_distribution_test(pvalue: float):
    """Evaluate whether H0 can be rejected"""
    if pvalue > 0.05:
        print(f"p={pvalue:.4f}: Data ostensibly has Gaussian distribution")
    else:
        print(f"p={pvalue:.4f}: Data ostensibly does NOT have Gaussian distribution")


def _print_normaltest(f: pd.Series) -> None:
    """Print SciPy NormalTest Result"""
    normaltest_result = normaltest(f, nan_policy="omit")
    print(f"Normal Test (Z-Score, P-Value): {normaltest_result.statistic:.4f}, {normaltest_result.pvalue:.4f}")
    _print_pass_normal_distribution_test(normaltest_result.pvalue)


def _print_jarque_bera(f: pd.Series) -> None:
    """Print SciPy Jarque-Bera Result"""
    jarque_bera_test_result = jarque_bera(f, nan_policy="omit")
    print(f"Jarque-Bera Test (Z-Score, P-Value): {jarque_bera_test_result.statistic:.4f}, {jarque_bera_test_result.pvalue:.4f}")
    _print_pass_normal_distribution_test(jarque_bera_test_result.pvalue)


def _plot(series: pd.Series, label: str) -> None:
    """Plot Univariate Graphs for Continuous Series"""
    figsize: tuple[int, int] = (8, 8)
    layout: tuple[int, int] = (8, 8)

    _ = series.hist(figsize=figsize)
    _ = plt.title(f"{label} (m^2) Distribution")
    plt.show()

    _ = series.plot(kind="density", sharex=False, figsize=figsize)
    _ = plt.title(f"{label} (m^2) Distribution")
    plt.show()

    _ = series.plot(kind="box", layout=layout, sharex=False, sharey=False, figsize=figsize)
    _ = plt.title(f"{label} (m^2) Box Plot")
    plt.show()

    _, floor_area_axis = plt.subplots(figsize=figsize)
    _ = probplot(series, dist=norm, plot=floor_area_axis)  # pyright: ignore[reportArgumentType]
    _ = floor_area_axis.set_title(f"{label} (m^2) Probability Plot")
    plt.show()


def continuous_data_analysis(x_train: gpd.GeoDataFrame) -> pd.Series:
    """Analyse the continuous data within the aggregated applications dataset"""

    floor_area_feature: pd.Series = x_train[_CONTINUOUS_DATA_FIELDS[0]]

    print("Floor Area M^2 Univariate Analysis:")
    print("----------------------------------")
    print(f"Missing Count: {floor_area_feature.isna().sum()}")
    print(f"Missing Rate: {floor_area_feature.isna().mean():.2f}")
    print("----------------------------------")
    print(f"Interquartile Range (IQR): {floor_area_feature.quantile(0.75) - floor_area_feature.quantile(0.25):.2f}")
    print(f"Median Absolute Deviation (MAD): {median_abs_deviation(floor_area_feature, nan_policy='omit'):.2f}")
    print(f"Skew: {skew(floor_area_feature, nan_policy='omit'):.2f}")
    print(f"Kurtosis: {kurtosis(floor_area_feature, nan_policy='omit'):.2f}")
    print(f"Standard Deviation.: {floor_area_feature.std():.2f}")
    print(f"Mean: {floor_area_feature.mean():.2f}")
    print(f"Median: {floor_area_feature.median():.2f}")
    print(f"Trimmed Mean (10%): {trim_mean(floor_area_feature, 0.1, nan_policy='omit'):.2f}")
    _print_normaltest(floor_area_feature)
    _print_jarque_bera(floor_area_feature)

    # Evaluate key statistics of trimmed distribution without outlier ranges (top and bottom 10%).
    trimmed_floor_area_feature = floor_area_feature[
        ((floor_area_feature >= floor_area_feature.quantile(0.1)) & (floor_area_feature <= floor_area_feature.quantile(0.9)))
    ]

    print("\nInterdecile Range (10%) Analysis:")
    print("-------------")
    print(f"Skew: {skew(trimmed_floor_area_feature, nan_policy='omit'):.2f}")
    print(f"Kurtosis: {kurtosis(trimmed_floor_area_feature, nan_policy='omit'):.2f}")
    print(f"S.D: {trimmed_floor_area_feature.std():.2f}")
    _print_normaltest(trimmed_floor_area_feature)
    _print_jarque_bera(trimmed_floor_area_feature)

    print(floor_area_feature.quantile([0.1, 0.25, 0.5, 0.75, 0.90]))

    print("\nDistribution Visual Analysis:")
    print("-------------")
    _plot(floor_area_feature, "Floor Area")

    # Apply Box-Cox power transform to maximise normality, as the underlying data demonstrates very high kurtosis and skew
    # (but the data does not have an exponential nature - it is a linear measure of floor area).

    # BoxCox and Log transform must be on positive values only.
    print(f"Minimum Floor Area (m^2): {floor_area_feature.min()}")
    floor_area_feature_complete: pd.Series = floor_area_feature.dropna()
    boxcox_floor_area_feature, boxcox_floor_area_lambda = boxcox(floor_area_feature_complete)
    boxcox_floor_area_feature_series: pd.Series = pd.Series(boxcox_floor_area_feature)

    print(f"Skew: {skew(boxcox_floor_area_feature_series):.2f}")
    print(f"Kurtosis: {kurtosis(boxcox_floor_area_feature_series):.2f}")
    # # Close to 0 - BoxCox converged closely to a Log transform
    print(f"BoxCox Lambda: {boxcox_floor_area_lambda:.2f}")
    #
    _print_normaltest(boxcox_floor_area_feature_series)
    _print_jarque_bera(boxcox_floor_area_feature_series)
    _plot(boxcox_floor_area_feature_series, "Box-Cox Transform Floor Area")

    return boxcox_floor_area_feature_series
