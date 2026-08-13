from pathlib import Path
from typing import Final

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pyproj.crs import CRS

LSOA_GPKG_FILE: Final[Path] = (
    Path(__file__).parent
    / "data"
    / "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5_-6970154227154374572.gpkg"
)

COUNCIL_GPKG_FILE: Final[Path] = (
    Path(__file__).parent
    / "data"
    / "Counties_and_Unitary_Authorities_December_2025_Boundaries_UK_BFC_-6415623133542248626.gpkg"
)

LSOA_IMD_CLASSIFICATION_FILE: Final[Path] = (
    Path(__file__).parent / "data" / "lsoa_imd_classification_2021.csv"
)

CONSERVATION_AREA_FILE: Final[Path] = Path(__file__).parent / "data" / "conservation_area.csv"


def _draw_gdf_lsoa_plot(
    applications_df: gpd.GeoDataFrame,
    lsoa_df: gpd.GeoDataFrame | None,
    councils_df: gpd.GeoDataFrame,
    title: str = "",
) -> None:
    """Plot the Application (GDF) Points over the LSOA Polygons"""

    _, ax = plt.subplots(figsize=(12, 12))

    if lsoa_df is not None:
        _ = lsoa_df.plot(
            ax=ax,
            column="Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOA",
            edgecolor="none",
            legend=True,
            cmap="Blues",
            vmin=1,
            vmax=10,
            linewidth=0.5,
        )

    _ = councils_df.plot(
        ax=ax,
        markersize=10,
        facecolor="none",
        edgecolor="black",
        alpha=0.7,
    )

    _ = applications_df.plot(
        ax=ax,
        markersize=10,
        facecolor="red",
        edgecolor="red",
        alpha=0.2,
    )

    cx.add_basemap(
        ax,
        source=cx.providers.CartoDB.Positron,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        zoom=12,  # pyright: ignore[reportArgumentType]
    )

    if lsoa_df is not None:
        minx, miny, maxx, maxy = lsoa_df.total_bounds

        xrange: float = maxx - minx
        yrange: float = maxy - miny
        padding: float = 0.05  # 5%

        _ = ax.set_xlim(minx - xrange * padding, maxx + xrange * padding)
        _ = ax.set_ylim(miny - yrange * padding, maxy + yrange * padding)

    ax.set_axis_off()

    if title:
        _ = ax.set_title(title)

    plt.show()


def _get_applications_gdf(applications_df: pd.DataFrame) -> tuple[gpd.GeoDataFrame, CRS]:
    """Get Applications GDF"""
    applications_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        applications_df.copy(),
        geometry=gpd.GeoSeries.from_wkt(applications_df["point"]),
        crs="EPSG:4326",
    )

    applications_gdf_crs: CRS | None = applications_gdf.crs

    if not applications_gdf_crs:
        raise ValueError("Unknown Applications CRS")

    return applications_gdf, applications_gdf_crs


def _get_lsoa_gdf(applications_gdf_crs: CRS) -> gpd.GeoDataFrame:
    """Get LSOAs GDF"""
    lsoa_gdf: gpd.GeoDataFrame = gpd.read_file(LSOA_GPKG_FILE)
    lsoa_gdf_crs: CRS | None = lsoa_gdf.crs

    if not lsoa_gdf_crs:
        raise ValueError("Unknown LSOA CRS")

    return lsoa_gdf.to_crs(applications_gdf_crs)


def _get_lsoa_imd_df() -> pd.DataFrame:
    """Get LSOAs IMD DF"""
    lsoa_imd_df: pd.DataFrame = pd.read_csv(
        LSOA_IMD_CLASSIFICATION_FILE,
        usecols=[
            "Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)",
            "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOA",
            "LSOA code (2021)",
            "LSOA name (2021)",
        ],
        dtype={
            "Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)": int,
            "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOA": int,
            "LSOA code (2021)": "string",
            "LSOA name (2021)": "string",
        },
        keep_default_na=False,
    )

    lsoa_imd_df.drop(
        columns=["Local Authority District code (2024)", "Local Authority District name (2024)"],
        inplace=True,
    )

    return lsoa_imd_df


def _get_councils_gdf(applications_gdf_crs: CRS) -> gpd.GeoDataFrame:
    """Get Councils GDF"""
    councils_gdf: gpd.GeoDataFrame = gpd.read_file(COUNCIL_GPKG_FILE)
    councils_gdf_crf: CRS | None = councils_gdf.crs

    if not councils_gdf_crf:
        raise ValueError("Unknown Councils CRS")

    return councils_gdf.to_crs(applications_gdf_crs)


def _get_conservation_areas_gdf(applications_gdf_crs: CRS) -> gpd.GeoDataFrame:
    """Get Conservation Areas GDF"""
    conservation_areas_df: pd.DataFrame = pd.read_csv(
        CONSERVATION_AREA_FILE,
        usecols=[
            "entity",
            "geometry",
            "name",
            "quality",
        ],
        dtype={
            "entity": "string",
            "geometry": "string",
            "name": "string",
            "quality": "categorical",
        },
        keep_default_na=False,
    )
    conservation_areas_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        conservation_areas_df.copy(),
        geometry=gpd.GeoSeries.from_wkt(conservation_areas_df["geometry"]),
        crs="EPSG:4326",
    )
    conservation_areas_gdf_crs: CRS | None = conservation_areas_gdf.crs

    if not conservation_areas_gdf_crs:
        raise ValueError("Unknown Conservation Areas CRS")

    return conservation_areas_gdf.to_crs(applications_gdf_crs)


def _merge_councils_and_lsoa(
    applications_gdf: gpd.GeoDataFrame, councils_gdf: gpd.GeoDataFrame, lsoa_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Merge Councils and LSOA Data with Applications"""
    applications_gdf = gpd.sjoin(
        applications_gdf,
        lsoa_gdf[["LSOA21CD", "LSOA21NM", "geometry"]],
        how="left",
        predicate="within",
    )
    applications_gdf = gpd.sjoin(
        applications_gdf,
        councils_gdf[
            [
                "CTYUA25CD",
                "CTYUA25NM",
                "geometry",
            ]
        ],
        how="left",
        predicate="within",
        rsuffix="_council",
    )

    print(f"N/A LSOAs: {applications_gdf['LSOA21CD'].isna().sum()}")

    return applications_gdf


def _merge_lsoa_imd(
    applications_gdf: gpd.GeoDataFrame, lsoa_gdf: gpd.GeoDataFrame, lsoa_imd_df: pd.DataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Merge LSOAs IMD with Applications, and with LSOAs"""
    applications_gdf = applications_gdf.merge(
        lsoa_imd_df, left_on="LSOA21CD", right_on="LSOA code (2021)"
    )
    lsoa_gdf = lsoa_gdf.merge(lsoa_imd_df, left_on="LSOA21CD", right_on="LSOA code (2021)")

    applications_gdf.drop(
        columns=["LSOA code (2021)", "LSOA name (2021)"], inplace=True
    )  # Duplicates after merge

    print(f"N/A LSOAs after IMD merge: {applications_gdf['LSOA21CD'].isna().sum()}")

    # Remove unknown LSOA observations as shown in initial plot
    applications_gdf.dropna(subset=["index_right"], inplace=True)

    return applications_gdf, lsoa_gdf


def _plot_lsoa(
    council: str,
    applications_gdf: gpd.GeoDataFrame,
    lsoa_gdf: gpd.GeoDataFrame,
    councils_gdf: gpd.GeoDataFrame,
) -> None:
    """Plot Individual LSOA and Contextual Data (Applications, Council, Conservation Area)"""
    council_subset_gdf: gpd.GeoDataFrame = councils_gdf[councils_gdf["CTYUA25NM"] == council]
    applications_subset_gdf: gpd.GeoDataFrame = applications_gdf[
        applications_gdf["CTYUA25NM"] == council
    ]

    lsoa_codes = applications_subset_gdf["LSOA21CD"].dropna().unique()
    lsoa_subset: gpd.GeoDataFrame = lsoa_gdf[lsoa_gdf["LSOA21CD"].isin(lsoa_codes)]

    _draw_gdf_lsoa_plot(
        applications_subset_gdf,
        lsoa_subset,
        council_subset_gdf,
        title=f"{council} ({council_subset_gdf['CTYUA25CD'].iloc[0]})",
    )


def preprocess_aggregate_geodata(df: DataFrame) -> gpd.GeoDataFrame:
    """Aggregate geodata data sources using LSOA boundary polygons"""

    applications_gdf, applications_crs = _get_applications_gdf(df)
    lsoa_gdf = _get_lsoa_gdf(applications_crs)
    lsoa_imd_df = _get_lsoa_imd_df()
    councils_gdf = _get_councils_gdf(applications_crs)
    conservation_areas_gdf = _get_conservation_areas_gdf(applications_crs)

    applications_gdf = _merge_councils_and_lsoa(applications_gdf, councils_gdf, lsoa_gdf)
    _draw_gdf_lsoa_plot(applications_gdf, None, councils_gdf, title="All Applications")

    applications_gdf, lsoa_gdf = _merge_lsoa_imd(applications_gdf, lsoa_gdf, lsoa_imd_df)
    _draw_gdf_lsoa_plot(
        applications_gdf,
        lsoa_gdf,
        councils_gdf,
        title="Applications within Valid LSOAs",
    )

    print(applications_gdf.groupby("CTYUA25NM").size())

    for council in applications_gdf["CTYUA25NM"].unique():
        _plot_lsoa(str(council), applications_gdf, lsoa_gdf, councils_gdf)

    # Group rare councils into 'Other' Category

    return applications_gdf
