"""Supplementary Preprocessed Geospatial Heritage Data"""

from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
from pyproj import CRS

_HERITAGE_LIST_FILE: Final[Path] = (
    Path(__file__).parent.parent / "data" / "heritage" / "National_Heritage_List_for_England_NHLE_v02_VIEW_4825054924992358738.gpkg"
)

EPSG_27700_CRS: Final[str] = "EPSG:27700"


def _map_listing_grade(value: object) -> int:
    """Map Raw Listing Grade Value to Ordinal Category"""
    if not value or (isinstance(value, float) and np.isnan(value)):
        return 0
    elif value == "II":
        return 3
    elif value == "II*":
        return 2
    elif value == "I":
        return 1
    else:
        raise ValueError(f"Unknown Listing value: {value}")


def preprocess_heritage_data(applications_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Preprocess and Append GeoSpatial Heritage Data"""

    applications_cdf_crs: CRS | None = applications_gdf.crs

    if not applications_cdf_crs:
        raise ValueError("Unknown Applications CRS")

    applications_gdf_epsg27700: gpd.GeoDataFrame = applications_gdf.to_crs(EPSG_27700_CRS)
    heritage_list_gdf: gpd.GeoDataFrame = gpd.read_file(
        _HERITAGE_LIST_FILE,
        date_format="%Y-%m-%d, %H:%M",
        parse_dates=["ListDate"],
        dtype={"Grade": "category", "Easting": int, "Northing": int},
        usecols=["Grade", "Easting", "Northing", "ListDate"],
    )

    heritage_list_gdf_epsg27700 = gpd.GeoDataFrame(
        heritage_list_gdf,
        geometry=gpd.points_from_xy(heritage_list_gdf["Easting"], heritage_list_gdf["Northing"]),
        crs=EPSG_27700_CRS,
    )

    heritage_list_gdf_epsg27700_crs: CRS | None = heritage_list_gdf_epsg27700.crs

    if not heritage_list_gdf_epsg27700_crs:
        raise ValueError("Unknown Heritage List CRS")

    applications_gdf = gpd.sjoin_nearest(
        applications_gdf_epsg27700,
        heritage_list_gdf_epsg27700[["Grade", "ListDate", "geometry"]],
        how="left",
        distance_col="distance_m",
        max_distance=1,
        rsuffix="_heritage_list",
    )

    applications_gdf["ListDate"] = applications_gdf["ListDate"].dt.tz_localize(None)

    applications_gdf["Grade"] = np.where(
        applications_gdf["ListDate"] <= applications_gdf["Decision Date"], applications_gdf["Grade"], np.nan
    )  # An approximation, as the application date is unavailable
    applications_gdf["Listing Grade"] = applications_gdf["Grade"].apply(_map_listing_grade)

    applications_gdf.drop(columns=["Grade", "ListDate"], inplace=True)

    print(f"{applications_gdf.groupby('Listing Grade').size()}\n")

    return applications_gdf.to_crs(applications_cdf_crs)
