"""Supplementary Preprocessed Geospatial Environmental Data"""

from pathlib import Path
from typing import Final

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import CRS

from csm500_project.constant import NO_PLOTS
from csm500_project.util import get_plot_lims, is_not_empty

_DATA_PATH = Path(__file__).parent.parent
_WOODLANDS_FILE: Final[Path] = _DATA_PATH / "data" / "environmental" / "Ancient_Woodland_England_-5278492453441969325.gpkg"
_FLOOD_RISK_FILE_PART_1: Final[Path] = _DATA_PATH / "data" / "environmental" / "flood-risk-zone.part01.csv"
_FLOOD_RISK_FILE_PART_2: Final[Path] = _DATA_PATH / "data" / "environmental" / "flood-risk-zone.part02.csv"
_GREEN_BELT_FILE: Final[Path] = _DATA_PATH / "data" / "environmental" / "green-belt.csv"
_GEOHAZARDS_FILE: Final[Path] = _DATA_PATH / "data" / "environmental" / "GB_Hex_5km_GeoSure_All_V8.gpkg"


def _get_flood_risk_gdf(applications_gdf_crs: CRS) -> gpd.GeoDataFrame:
    """Load, combine flood file risk files and return GDF"""

    cols: list[str] = ["geometry", "flood-risk-level", "flood-risk-type"]
    dtype: dict[str, str] = {"geometry": "string", "flood-risk-level": "category", "flood-risk-type": "category"}

    flood_risk_df_1: pd.DataFrame = pd.read_csv(_FLOOD_RISK_FILE_PART_1, usecols=cols, dtype=dtype, keep_default_na=False)
    flood_risk_df_2: pd.DataFrame = pd.read_csv(_FLOOD_RISK_FILE_PART_2, usecols=cols, dtype=dtype, keep_default_na=False)
    flood_risk_df: pd.DataFrame = pd.concat([flood_risk_df_1, flood_risk_df_2], ignore_index=True)
    flood_risk_df = flood_risk_df[flood_risk_df["geometry"].apply(is_not_empty)]

    return gpd.GeoDataFrame(
        flood_risk_df,
        geometry=gpd.GeoSeries.from_wkt(flood_risk_df["geometry"], on_invalid="warn"),
        crs="EPSG:4326",
    ).to_crs(applications_gdf_crs)


def _get_geohazards_gdfs(applications_gdf_crs: CRS) -> dict[str, gpd.GeoDataFrame]:
    """Get Geohazards GDF"""

    layers = gpd.list_layers(_GEOHAZARDS_FILE)

    print("Geohazards Layers:", layers)

    geohazard_gdfs: dict[str, gpd.GeoDataFrame] = {}
    geohazards_layer_map: dict[str, str] = {
        "GB_Hex_5km_GS_CollapsibleDeposits_v8": "Geohazards Collapsible Deposits",
        "GB_Hex_5km_GS_CompressibleGround_v8": "Geohazards Compressible Ground",
        "GB_Hex_5km_GS_Landslides_v8": "Geohazards Landslides",
        "GB_Hex_5km_GS_RunningSand_v8": "Geohazards Running Sand",
        "GB_Hex_5km_GS_ShrinkSwell_v8": "Geohazards Shrink Swell",
        "GB_Hex_5km_GS_SolubleRocks_v8": "Geohazards Soluble Rocks",
    }

    for layer_name in layers["name"]:
        if layer_name in geohazards_layer_map:
            layer_gdf = gpd.read_file(_GEOHAZARDS_FILE, layer=layer_name)

            geohazard_gdfs[geohazards_layer_map[layer_name]] = layer_gdf.to_crs(applications_gdf_crs)

    for layer, gdf in geohazard_gdfs.items():
        if not gdf.crs:
            raise ValueError(f"Unknown Geohazards Layer CRS, {layer}: {gdf.crs}")

    return geohazard_gdfs


def _get_woodlands_gdf(applications_gdf_crs: CRS) -> gpd.GeoDataFrame:
    """Get Woodlands GDF"""
    woodlands_gdf: gpd.GeoDataFrame = gpd.read_file(_WOODLANDS_FILE)
    woodlands_gdf_crs: CRS | None = woodlands_gdf.crs

    if not woodlands_gdf_crs:
        raise ValueError("Unknown Woodlands CRS")

    return woodlands_gdf.to_crs(applications_gdf_crs)


def _get_green_belt_gdf(applications_gdf_crs: CRS) -> gpd.GeoDataFrame:
    """Load and create the Green Belt GDF"""
    green_belt_df: pd.DataFrame = pd.read_csv(_GREEN_BELT_FILE, usecols=["geometry"], dtype={"geometry": "string"}, keep_default_na=False)
    green_belt_df = green_belt_df[green_belt_df["geometry"].apply(is_not_empty)]

    return gpd.GeoDataFrame(
        green_belt_df,
        geometry=gpd.GeoSeries.from_wkt(green_belt_df["geometry"], on_invalid="warn"),
        crs="EPSG:4326",
    ).to_crs(applications_gdf_crs)


def _is_within_area(x):
    return (x.notna().any()).astype(int)


def _plot_environmental_maps(
    applications_gdf: gpd.GeoDataFrame,
    environmental_gdf: gpd.GeoDataFrame | None,
    hazard_environmental_gdf: gpd.GeoDataFrame | None,
    title: str = "",
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
) -> None:
    """Plot Environmental Data Against Applications"""
    if NO_PLOTS:
        return

    _, ax = plt.subplots(figsize=(12, 12))

    if environmental_gdf is not None:
        _ = environmental_gdf.plot(
            ax=ax,
            markersize=5,
            facecolor="yellow",
            edgecolor="none",
            alpha=0.7,
        )

    if hazard_environmental_gdf is not None:
        _ = hazard_environmental_gdf.plot(
            ax=ax,
            column="CLASS",
            edgecolor="none",
            legend=True,
            cmap="Blues",
            linewidth=0.5,
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


def preprocess_environmental_data(applications_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Preprocess and Append GeoSpatial Environmental Data"""

    applications_gdf_crs: CRS | None = applications_gdf.crs

    if not applications_gdf_crs:
        raise ValueError("Unknown applications CRS")

    geohazards_gdfs: dict[str, gpd.GeoDataFrame] = _get_geohazards_gdfs(applications_gdf_crs)
    flood_risk_gdf: gpd.GeoDataFrame = _get_flood_risk_gdf(applications_gdf_crs)
    green_belt_gdf: gpd.GeoDataFrame = _get_green_belt_gdf(applications_gdf_crs)
    woodlands_gdf: gpd.GeoDataFrame = _get_woodlands_gdf(applications_gdf_crs)

    applications_gdf = gpd.sjoin(
        applications_gdf,
        flood_risk_gdf[["flood-risk-level", "flood-risk-type", "geometry"]],
        how="left",
        predicate="within",
        rsuffix="flood_risk",
    )

    applications_gdf = gpd.sjoin(
        applications_gdf,
        green_belt_gdf[["geometry"]],
        how="left",
        predicate="within",
        rsuffix="green_belt",
    )

    applications_gdf = gpd.sjoin(
        applications_gdf,
        woodlands_gdf[["geometry"]],
        how="left",
        predicate="within",
        rsuffix="woodlands",
    )

    for layer, gdf in geohazards_gdfs.items():
        applications_gdf = gpd.sjoin(
            applications_gdf,
            gdf[["geometry"]],
            how="left",
            predicate="within",
            rsuffix=layer,
        )

    applications_gdf["In Flood Zone"] = applications_gdf.groupby(level=0)["index_flood_risk"].apply(_is_within_area)
    print(f"Applications within Flood Zone\n {applications_gdf['In Flood Zone'].value_counts()}")
    _plot_environmental_maps(applications_gdf, flood_risk_gdf, None, "Applications Within Flood Zones", *get_plot_lims(applications_gdf))

    applications_gdf["In Green Belt"] = applications_gdf.groupby(level=0)["index_green_belt"].apply(_is_within_area)
    print(f"Applications within Green Belt\n {applications_gdf['In Green Belt'].value_counts()}")
    _plot_environmental_maps(applications_gdf, green_belt_gdf, None, "Applications Within Green Belts", *get_plot_lims(applications_gdf))

    applications_gdf["In Ancient Woodlands"] = applications_gdf.groupby(level=0)["index_woodlands"].apply(_is_within_area)
    print(f"Applications within Ancient Woodlands\n {applications_gdf['In Ancient Woodlands'].value_counts()}")
    _plot_environmental_maps(
        applications_gdf, woodlands_gdf, None, "Applications Within Ancient Woodlands", *get_plot_lims(applications_gdf)
    )

    for layer, geohazard_gdf in geohazards_gdfs.items():
        _plot_environmental_maps(applications_gdf, None, geohazard_gdf, f"Applications Within {layer}", *get_plot_lims(geohazard_gdf))

    return applications_gdf
