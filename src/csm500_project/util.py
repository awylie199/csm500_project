"""Project Utilities"""

import geopandas as gpd


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
