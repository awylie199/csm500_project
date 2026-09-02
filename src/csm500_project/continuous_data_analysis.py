"""Continuous Data Analysis"""

import geopandas as gpd

_CONTINUOUS_DATA_FIELDS: tuple[str, ...] = ("Floor Area (m^2)",)


def continuous_data_analysis(x_train: gpd.GeoDataFrame) -> None:
    """Analyse the continuous data within the aggregated applications dataset"""
