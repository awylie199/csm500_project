"""Supplementary Preprocessed Geospatial Environmental Data"""

from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

_DATA_PATH = Path(__file__).parent.parent

_WOODLANDS_FILE: Final[Path] = _DATA_PATH / "data" / "environmental" / "Ancient_Woodland_England_-5278492453441969325.gpkg"
_FLOOD_RISK_FILE_PART_1: Final[Path] = _DATA_PATH / "data" / "environmental" / "flood-risk-zone.part01.csv"
_FLOOD_RISK_FILE_PART_2: Final[Path] = _DATA_PATH / "data" / "environmental" / "flood-risk-zone.part02.csv"
_GREEN_BELT_FILE: Final[Path] = _DATA_PATH / "data" / "environmental" / "green-belt.csv"
_GEOHAZARDS_FILE: Final[Path] = _DATA_PATH / "data" / "environmental" / "GB_Hex_5km_GeoSure_All_V8.gpkg"


def _combine_flood_risk_files() -> pd.DataFrame:
    """Combine flood file risk files"""

    cols: list[str] = ["geometry", "flood-risk-level", "flood-risk-type"]
    dtype: dict[str, str] = {"geometry": "string", "flood-risk-level": "category", "flood-risk-type": "category"}

    flood_risk_gdf_1 = pd.read_csv(_FLOOD_RISK_FILE_PART_1, usecols=cols, dtype=dtype, keep_default_na=False)
    flood_risk_gdf_2 = pd.read_csv(_FLOOD_RISK_FILE_PART_2, usecols=cols, dtype=dtype, keep_default_na=False)

    return pd.concat([flood_risk_gdf_1, flood_risk_gdf_2], ignore_index=True)


def _get_geohazards_gdf() -> gpd.GeoDataFrame:
    """Get Geohazards GDF"""
    geohazards_gdf = gpd.GeoDataFrame = gpd.read_file(_GEOHAZARDS_FILE)
    geohazards_gdf_crs: CRS | None = geohazards_gdf.crs

    if not geohazards_gdf_crs:
        raise ValueError("Unknown Geohazards CRS")

    return geohazards_gdf


def _get_woodlands_gdf() -> gpd.GeoDataFrame:
    """Get Woodlands GDF"""
    woodlands_gdf = gpd.GeoDataFrame = gpd.read_file(_WOODLANDS_FILE)
    woodlands_gdf_crs: CRS | None = woodlands_gdf.crs

    if not woodlands_gdf_crs:
        raise ValueError("Unknown Woodlands CRS")

    return woodlands_gdf


def preprocess_environmental_data(applications_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Preprocess and Append GeoSpatial Environmental Data"""

    flood_risk_df: pd.DataFrame = _combine_flood_risk_files()
    green_belt_df: pd.DataFrame = pd.read_csv(_GREEN_BELT_FILE, usecols=["geometry"], dtype={"geometry": "string"}, keep_default_na=False)
    geohazards_gdf: gpd.GeoDataFrame = _get_geohazards_gdf()
    woodlands_gdf: gpd.GeoDataFrame = _get_woodlands_gdf()
