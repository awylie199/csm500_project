"""Project Utilities"""

import geopandas as gpd
import pandas as pd


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
    x_df: pd.DataFrame | gpd.GeoDataFrame, y_df: pd.DataFrame | gpd.GeoDataFrame
) -> dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    physical_cols: list[str] = ["Property Type", "Is Domestic", "Floor Area (m^2)"]
    not_na = x_df[physical_cols].notna().all(axis=1)

    return {
        "x_primary": x_df.copy(deep=True).drop(columns=physical_cols),
        "x_physical": x_df.copy(deep=True).loc[not_na],
        "y_primary": y_df.copy(deep=True),
        "y_physical": y_df.copy(deep=True).loc[not_na],
    }
