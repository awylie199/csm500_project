"""Supplementary Preprocessed Land Ownership Heritage Data"""

import os
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from scipy.spatial import cKDTree

from csm500_project.constant import EPSG_27700_CRS, USE_UPRN_CACHE

_GRANDPARENT_PATH: Final[Path] = Path(__file__).parent.parent
_DATA_PATH: Final[Path] = _GRANDPARENT_PATH / "data" / "physical"
_COMMERCIAL_CERTS_FILE: Final[Path] = _DATA_PATH / "commercial_energy_certs.csv"
_DOMESTIC_CERTS_FILE: Final[Path] = _DATA_PATH / "domestic_energy_certs.csv"
_UPRN_FILE_PART_1: Final[Path] = _DATA_PATH / "osopenuprn_202608.part01.csv"
_UPRN_FILE_PART_2: Final[Path] = _DATA_PATH / "osopenuprn_202608.part02.csv"
_UPRN_FILE_PART_3: Final[Path] = _DATA_PATH / "osopenuprn_202608.part03.csv"
_APPLICATIONS_WITH_UPRN_PARQUET_PATH = _GRANDPARENT_PATH / "cache" / "applications_with_uprn.parquet"


def _get_uprn_gdf() -> gpd.GeoDataFrame:
    """Load and Join UPRN Data to Aggregated GDF"""

    cols: list[str] = ["UPRN", "X_COORDINATE", "Y_COORDINATE"]
    dtype: dict[str, str] = {"UPRN": "string", "X_COORDINATE": "float64", "Y_COORDINATE": "float64"}

    uprn_df_1: pd.DataFrame = pd.read_csv(_UPRN_FILE_PART_1, usecols=cols, dtype=dtype, keep_default_na=False)
    uprn_df_2: pd.DataFrame = pd.read_csv(_UPRN_FILE_PART_2, usecols=cols, dtype=dtype, keep_default_na=False)
    uprn_df_3: pd.DataFrame = pd.read_csv(_UPRN_FILE_PART_3, usecols=cols, dtype=dtype, keep_default_na=False)
    uprn_df: pd.DataFrame = pd.concat([uprn_df_1, uprn_df_2, uprn_df_3], ignore_index=True)

    return gpd.GeoDataFrame(
        uprn_df,
        geometry=gpd.points_from_xy(uprn_df["X_COORDINATE"], uprn_df["Y_COORDINATE"]),
        crs=EPSG_27700_CRS,
    )


def _get_certs_df() -> pd.DataFrame:
    """Load and Return Combined Certs (Physical Properties) DF"""

    cols: list[str] = ["uprn", "property_type"]
    dtype: dict[str, str] = {"uprn": "string", "property_type": "category"}

    domestic_df: pd.DataFrame = pd.read_csv(
        _DOMESTIC_CERTS_FILE, usecols=[*cols, "total_floor_area"], dtype={**dtype, "total_floor_area": int}, keep_default_na=False
    )
    domestic_df.rename(columns={"total_floor_area": "Floor Area (m^2)"}, inplace=True)

    commercial_df: pd.DataFrame = pd.read_csv(
        _COMMERCIAL_CERTS_FILE, usecols=[*cols, "floor_area"], dtype={**dtype, "floor_area": int}, keep_default_na=False
    )
    commercial_df.rename(columns={"floor_area": "Floor Area (m^2)"}, inplace=True)
    domestic_df["Is Domestic"] = 1
    commercial_df["Is Domestic"] = 0

    return pd.concat([domestic_df, commercial_df], ignore_index=True).rename(columns={"property_type": "Property Type"})


def _check_join(applications_with_uprn: gpd.GeoDataFrame, applications_gdf: gpd.GeoDataFrame):
    print(f"Original Input Applications: {len(applications_gdf)}")
    print(f"Final Output DataFrame:     {len(applications_with_uprn)}")
    print(f"Successfully Matched:       {applications_with_uprn['UPRN'].notna().sum()}")
    print(f"Unmatched (Kept as NaN):    {applications_with_uprn['UPRN'].isna().sum()}")

    assert len(applications_with_uprn) == len(applications_gdf)


def join_applications_to_uprn(
    applications_gdf: gpd.GeoDataFrame,
    uprn_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    applications_pos_id: Final[str] = "_application_pos"
    uprn_pos_id: Final[str] = "uprn_pos"
    distance_id: Final[str] = "_distance_m"

    applications_gdf_copy: gpd.GeoDataFrame = applications_gdf.copy()
    uprn_gdf_copy: gpd.GeoDataFrame = uprn_gdf.copy()
    applications_gdf_copy[applications_pos_id] = np.arange(len(applications_gdf_copy))
    uprn_gdf_copy[uprn_pos_id] = np.arange(len(uprn_gdf_copy))

    applications_xy = np.column_stack(
        (
            applications_gdf_copy.geometry.x.to_numpy(),
            applications_gdf_copy.geometry.y.to_numpy(),
        )
    )
    uprn_xy = np.column_stack(
        (
            uprn_gdf_copy.geometry.x.to_numpy(),
            uprn_gdf_copy.geometry.y.to_numpy(),
        )
    )

    tree: cKDTree = cKDTree(uprn_xy)
    # query_ball_point type annotation is incorrect when arg[0] is a list
    candidate_application_positions: list[list[int]] = tree.query_ball_point(  # pyright: ignore[reportAssignmentType]
        applications_xy,
        r=10.0,
    )

    candidate_rows: list[tuple[int, int, float]] = []

    for application_position, uprn_positions in enumerate(candidate_application_positions):
        for uprn_position in uprn_positions:
            # Find the diagonal distance between X and Y. Scalar return value when x and y are scalars.
            distance: float = np.hypot(
                applications_xy[application_position, 0] - uprn_xy[uprn_position, 0],
                applications_xy[application_position, 1] - uprn_xy[uprn_position, 1],
            )

            candidate_rows.append(
                (
                    application_position,
                    uprn_position,
                    distance,
                )
            )

    candidates = pd.DataFrame(
        candidate_rows,
        columns=[
            applications_pos_id,
            uprn_pos_id,
            distance_id,
        ],
    )

    candidates = candidates.sort_values(
        distance_id,
        kind="stable",
    ).reset_index(drop=True)

    used_application_positions: set[int] = set()
    used_uprn_positions: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for application_position, uprn_position, distance in candidates[
        [
            applications_pos_id,
            uprn_pos_id,
            distance_id,
        ]
    ].itertuples(index=False, name=None):  # Stop renaming of columns starting with _
        application_position: int = int(application_position)
        uprn_position: int = int(uprn_position)

        if (application_position in used_application_positions) or (uprn_position in used_uprn_positions):
            continue

        used_application_positions.add(application_position)
        used_uprn_positions.add(uprn_position)

        matches.append(
            (
                application_position,
                uprn_position,
                distance,
            )
        )

    applications_with_positions_gdf: gpd.GeoDataFrame = applications_gdf_copy.merge(
        pd.DataFrame(
            matches,
            columns=[
                applications_pos_id,
                uprn_pos_id,
                distance_id,
            ],
        ),
        on=applications_pos_id,
        how="left",
    )

    applications_with_uprn_gdf: gpd.GeoDataFrame = applications_with_positions_gdf.merge(
        uprn_gdf_copy.drop(columns="geometry"),
        on=uprn_pos_id,
        how="left",
    )

    print(applications_with_uprn_gdf[distance_id].describe())

    return gpd.GeoDataFrame(
        applications_with_uprn_gdf.drop(columns=[applications_pos_id, uprn_pos_id, distance_id], errors="raise"),
        geometry="geometry",
        crs=applications_gdf_copy.crs,
    )


def preprocess_physical_data(applications_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Preprocess and Append GeoSpatial Physical Data Based on UPRN"""

    applications_cdf_crs: CRS | None = applications_gdf.crs

    if not applications_cdf_crs:
        raise ValueError("Unknown Applications CRS")

    uprn_gdf: gpd.GeoDataFrame = _get_uprn_gdf()
    uprn_gdf_crs: CRS | None = uprn_gdf.crs

    if not uprn_gdf_crs:
        raise ValueError("Unknown UPRN CRS")

    applications_gdf_epsg27700: gpd.GeoDataFrame = applications_gdf.to_crs(EPSG_27700_CRS)

    if USE_UPRN_CACHE and os.path.exists(_APPLICATIONS_WITH_UPRN_PARQUET_PATH):
        applications_gdf = gpd.read_parquet(_APPLICATIONS_WITH_UPRN_PARQUET_PATH)
        print("Loaded Applications with UPRN from Cache.")
    else:
        applications_gdf = join_applications_to_uprn(applications_gdf_epsg27700, uprn_gdf)
        _check_join(applications_gdf, applications_gdf_epsg27700)
        applications_gdf.to_parquet(_APPLICATIONS_WITH_UPRN_PARQUET_PATH, engine="pyarrow", compression="snappy")
        print("Saved Applications with UPRN to Cache.")

    return applications_gdf.merge(
        _get_certs_df(),
        left_on="UPRN",
        right_on="uprn",
        how="left",
    ).drop(columns=["X_COORDINATE", "Y_COORDINATE", "UPRN", "uprn"])
