from pathlib import Path
from typing import Final

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pyproj.crs import CRS

from .constant import NO_PLOTS
from .util import get_plot_lims, is_not_empty

LSOA_GPKG_FILE: Final[Path] = (
    Path(__file__).parent / "data" / "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5_-6970154227154374572.gpkg"
)

COUNCIL_GPKG_FILE: Final[Path] = (
    Path(__file__).parent / "data" / "Counties_and_Unitary_Authorities_December_2025_Boundaries_UK_BFC_-6415623133542248626.gpkg"
)

LSOA_IMD_CLASSIFICATION_FILE: Final[Path] = Path(__file__).parent / "data" / "lsoa_imd_classification_2021.csv"

CONSERVATION_AREA_FILE: Final[Path] = Path(__file__).parent / "data" / "conservation_area.csv"


def _draw_gdf_applications_plot(
    applications_gdf: gpd.GeoDataFrame,
    lsoa_df: gpd.GeoDataFrame | None,
    councils_gdf: gpd.GeoDataFrame,
    conservation_areas_gdf: gpd.GeoDataFrame | None,
    title: str = "",
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
) -> None:
    """Plot the Application (GDF) Points over the LSOA Polygons or Conservation Area Polygons"""

    if NO_PLOTS:
        return

    _, ax = plt.subplots(figsize=(12, 12))

    if conservation_areas_gdf is not None and not conservation_areas_gdf.empty:
        _ = conservation_areas_gdf.plot(
            ax=ax,
            markersize=5,
            facecolor="yellow",
            edgecolor="none",
            alpha=0.7,
        )

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

    _ = councils_gdf.plot(
        ax=ax,
        markersize=5,
        facecolor="none",
        edgecolor="black",
        alpha=0.7,
    )

    _ = applications_gdf.plot(
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

    if x_lim and y_lim:
        _ = ax.set_xlim(x_lim[0], x_lim[1])
        _ = ax.set_ylim(y_lim[0], y_lim[1])

    ax.set_axis_off()

    if title:
        _ = ax.set_title(title)

    plt.show()


def _get_applications_gdf(applications_df: pd.DataFrame) -> tuple[gpd.GeoDataFrame, CRS]:
    """Get Applications GDF"""
    applications_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        applications_df.copy(),
        geometry=gpd.GeoSeries.from_wkt(applications_df["Point"]),
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

    return lsoa_imd_df


def _get_councils_gdf(applications_gdf_crs: CRS) -> gpd.GeoDataFrame:
    """Get Councils GDF"""
    councils_gdf: gpd.GeoDataFrame = gpd.read_file(COUNCIL_GPKG_FILE)
    councils_gdf_crs: CRS | None = councils_gdf.crs

    if not councils_gdf_crs:
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
            "quality": "category",
        },
        keep_default_na=False,
    )

    missing_geometry = conservation_areas_df["geometry"].isna() | conservation_areas_df["geometry"].eq("")
    print(f"N/A Conservation Ares: {missing_geometry.sum()}")

    conservation_areas_df = conservation_areas_df[conservation_areas_df["geometry"].apply(is_not_empty)]
    conservation_areas_gdf = gpd.GeoDataFrame(
        conservation_areas_df,
        geometry=gpd.GeoSeries.from_wkt(conservation_areas_df["geometry"], on_invalid="warn"),
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
    applications_gdf = applications_gdf.merge(lsoa_imd_df, left_on="LSOA21CD", right_on="LSOA code (2021)")
    lsoa_gdf = lsoa_gdf.merge(lsoa_imd_df, left_on="LSOA21CD", right_on="LSOA code (2021)")

    applications_gdf.drop(columns=["LSOA code (2021)", "LSOA name (2021)"], inplace=True)  # Duplicates after merge

    print(f"N/A LSOAs after IMD merge: {applications_gdf['LSOA21CD'].isna().sum()}")

    # Remove unknown LSOA observations as shown in initial plot
    applications_gdf.dropna(subset=["index_right"], inplace=True)

    return applications_gdf, lsoa_gdf


def _plot_lsoa(
    council: str,
    applications_gdf: gpd.GeoDataFrame,
    lsoa_gdf: gpd.GeoDataFrame,
    councils_gdf: gpd.GeoDataFrame,
    conservation_areas_gdf: gpd.GeoDataFrame,
) -> None:
    """Plot Individual LSOA and Contextual Data (Applications, Councils, Conservation Areas)"""
    council_subset_gdf: gpd.GeoDataFrame = councils_gdf[councils_gdf["CTYUA25NM"] == council]
    applications_subset_gdf: gpd.GeoDataFrame = applications_gdf[applications_gdf["CTYUA25NM"] == council]

    lsoa_codes = applications_subset_gdf["LSOA21CD"].dropna().unique()
    lsoa_subset_gdf: gpd.GeoDataFrame = lsoa_gdf[lsoa_gdf["LSOA21CD"].isin(lsoa_codes)]

    conservation_areas_subset_gdf = gpd.sjoin(
        conservation_areas_gdf,
        lsoa_subset_gdf,
        how="inner",
        predicate="intersects",
        rsuffix="_conservation_area",
    )

    x_lim, y_lim = get_plot_lims(lsoa_subset_gdf)

    _draw_gdf_applications_plot(
        applications_subset_gdf,
        lsoa_subset_gdf,
        council_subset_gdf,
        None,
        title=f"{council} ({council_subset_gdf['CTYUA25CD'].iloc[0]}) LSOAs and Applications",
        x_lim=x_lim,
        y_lim=y_lim,
    )
    _draw_gdf_applications_plot(
        applications_subset_gdf,
        None,
        council_subset_gdf,
        conservation_areas_subset_gdf,
        title=f"{council} ({council_subset_gdf['CTYUA25CD'].iloc[0]}) Conservation Areas and Applications",
        x_lim=x_lim,
        y_lim=y_lim,
    )


def preprocess_aggregate_geodata(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Aggregate geodata data sources using LSOA boundary polygons"""

    applications_gdf, applications_crs = _get_applications_gdf(df)
    lsoa_gdf = _get_lsoa_gdf(applications_crs)
    lsoa_imd_df = _get_lsoa_imd_df()
    councils_gdf = _get_councils_gdf(applications_crs)
    conservation_areas_gdf = _get_conservation_areas_gdf(applications_crs)

    applications_gdf = _merge_councils_and_lsoa(applications_gdf, councils_gdf, lsoa_gdf)
    _draw_gdf_applications_plot(
        applications_gdf,
        None,
        councils_gdf,
        conservation_areas_gdf,
        title="All Applications and Conservation Areas",
    )

    applications_gdf, lsoa_gdf = _merge_lsoa_imd(applications_gdf, lsoa_gdf, lsoa_imd_df)
    _draw_gdf_applications_plot(
        applications_gdf,
        lsoa_gdf,
        councils_gdf,
        conservation_areas_gdf,
        title="Applications within Valid LSOAs",
    )

    print(applications_gdf.groupby("CTYUA25NM").size())

    for council in applications_gdf["CTYUA25NM"].unique():
        _plot_lsoa(str(council), applications_gdf, lsoa_gdf, councils_gdf, conservation_areas_gdf)

    applications_gdf_with_conservation_areas = gpd.sjoin(
        applications_gdf[["geometry"]],
        conservation_areas_gdf[["geometry"]],
        how="left",
        predicate="within",
        rsuffix="conservation_area",
    )

    # Applications may fall within overlapping conservation areas
    applications_gdf["In Conservation Area"] = (
        applications_gdf_with_conservation_areas.groupby(level=0)["index_conservation_area"].apply(lambda x: x.notna().any()).astype(int)
    )

    print(f"Applications within Conservation Areas\n {applications_gdf['In Conservation Area'].value_counts()}")

    # Percentage of Applications within each IMD decile
    print(
        f"Applications within IMD Deciles\n {applications_gdf['Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOA'].value_counts(sort=True)}"
    )

    # N.B. 4 year gap between LSOA and Council boundaries which may explain some discrepancies on map alignment
    applications_gdf.rename(
        columns={
            "Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)": "IMD Rank",
            "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOA": "IMD Decile",
            "LSOA21CD": "LSOA Code",
            "CTYUA25CD": "Council Code",
        },
        inplace=True,
    )

    return applications_gdf
