"""Supplementary Preprocessed Geospatial Administrative Data"""

import os
from pathlib import Path
from typing import Final

import dask_geopandas as dgpd
import datashader as ds
import datashader.transfer_functions as tf
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from geopandas.geoseries import GeoSeries
from matplotlib.lines import Line2D
from pyproj import CRS

from csm500_project.constant import EPSG_4326_CRS, EPSG_27700_CRS, NO_PLOTS
from csm500_project.util import get_plot_lims, is_within_area

_DATA_PATH = Path(__file__).parent.parent / "data" / "administrative"
_BROWNFIELDS_FILE: Final[Path] = _DATA_PATH / "brownfield-land.csv"
_BUILT_UP_AREAS_FILE: Final[Path] = _DATA_PATH / "BUA_2022_GB_8703850870315979820.gpkg"
_SPECIAL_AREAS_OF_CONSERVATION_FILE: Final[Path] = _DATA_PATH / "Special_Areas_of_Conservation_England_3848535579062436380.gpkg"
_PARISH_FILE: Final[Path] = _DATA_PATH / "PARNCP_MAY_2026_EW_BFC_-1299597603504015581.gpkg"
_ARTICLE_4_DIRECTION_FILE: Final[Path] = _DATA_PATH / "article-4-direction-area.csv"

_BROWNFIELD_START_DATE_COLUMN: Final[str] = "Brownfield Start Date"
_BROWNFIELD_END_DATE_COLUMN: Final[str] = "Brownfield End Date"
_ARTICLE_4_START_DATE_COLUMN: Final[str] = "Article 4 Start Date"
_ARTICLE_4_END_DATE_COLUMN: Final[str] = "Article 4 End Date"

_BROWNFIELDS_CATEGORY_FIELD: Final[str] = "Near Brownfield Site"
_BUILT_UP_AREAS_CATEGORY_FIELD: Final[str] = "In Built Up Area"
_SPECIAL_AREAS_OF_CONSERVATION_CATEGORY_FIELD: Final[str] = "In Special Area of Conservation"
_ARTICLE_4_DIRECTION_CATEGORY_FIELD: Final[str] = "In Article 4 Area"
_PARISH_CATEGORY_FIELD: Final[str] = "Parish Code"
_CATEGORY_FIELDS: Final[list[str]] = [
    _BROWNFIELDS_CATEGORY_FIELD,
    _BUILT_UP_AREAS_CATEGORY_FIELD,
    _SPECIAL_AREAS_OF_CONSERVATION_CATEGORY_FIELD,
    _ARTICLE_4_DIRECTION_CATEGORY_FIELD,
    _PARISH_CATEGORY_FIELD,
]

_APPLICATIONS_GDF_ID: Final[str] = "applications"

# Geometry columns are dropped on join - used to plot points subsequently
_BROWNFIELD_GDF_GEOMETRY_COLUMN: Final[str] = "brownfields_geometry"
_BUILT_UP_AREAS_GDF_GEOMETRY_COLUMN: Final[str] = "built_up_areas_geometry"
_SPECIAL_AREAS_OF_CONSERVATION_GEOMETRY_COLUMN: Final[str] = "special_areas_of_conservation_geometry"
_PARISH_GEOMETRY_COLUMN: Final[str] = "parish_geometry"
_ARTICLE_4_DIRECTION_GEOMETRY_COLUMN: Final[str] = "article_4_direction_geometry"


def _get_brownfields_gdf() -> gpd.GeoDataFrame:
    """Get Brownfields GDF"""

    brownfields_df: pd.DataFrame = pd.read_csv(
        _BROWNFIELDS_FILE,
        date_format="%Y-%m-%d",
        parse_dates=["start-date", "end-date"],
        usecols=["end-date", "start-date", "point"],
        dtype={"point": "string"},
        keep_default_na=False,
    )
    brownfields_df.dropna(subset=["point"], inplace=True)
    brownfields_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        brownfields_df, geometry=gpd.GeoSeries.from_wkt(brownfields_df["point"], on_invalid="fix"), crs=EPSG_4326_CRS
    )
    brownfields_gdf = brownfields_gdf.rename(
        columns={"start-date": _BROWNFIELD_START_DATE_COLUMN, "end-date": _BROWNFIELD_END_DATE_COLUMN}
    ).to_crs(EPSG_27700_CRS)
    brownfields_gdf[_BROWNFIELD_GDF_GEOMETRY_COLUMN] = brownfields_gdf.geometry

    return brownfields_gdf


def _get_built_up_areas_gdf() -> gpd.GeoDataFrame:
    """Get Built-Up Areas GDF"""

    built_up_areas_gdf: gpd.GeoDataFrame = gpd.read_file(
        _BUILT_UP_AREAS_FILE,
        dtype={
            "BNG_E": "float64",
            "BNG_N": "float64",
        },
        usecols=["BNG_E", "BNG_N"],
    )
    built_up_areas_gdf = gpd.GeoDataFrame(
        built_up_areas_gdf,
        crs=EPSG_27700_CRS,
    )
    built_up_areas_gdf[_BUILT_UP_AREAS_GDF_GEOMETRY_COLUMN] = built_up_areas_gdf.geometry

    return built_up_areas_gdf


def _get_special_areas_of_conservation_gdf() -> gpd.GeoDataFrame:
    """Get Special Areas of Conservation GDF"""

    special_areas_of_conservation_gdf: gpd.GeoDataFrame = gpd.read_file(
        _SPECIAL_AREAS_OF_CONSERVATION_FILE,
        dtype={"EASTING": "float64", "NORTHING": "float64"},
        usecols=["EASTING", "NORTHING"],
    )
    special_areas_of_conservation_gdf = gpd.GeoDataFrame(
        special_areas_of_conservation_gdf,
        crs=EPSG_27700_CRS,
    ).to_crs(EPSG_27700_CRS)
    special_areas_of_conservation_gdf[_SPECIAL_AREAS_OF_CONSERVATION_GEOMETRY_COLUMN] = special_areas_of_conservation_gdf.geometry

    return special_areas_of_conservation_gdf


def _get_parish_gdf() -> gpd.GeoDataFrame:
    """Get Parish GDF"""

    parish_gdf: gpd.GeoDataFrame = gpd.read_file(
        _PARISH_FILE,
        dtype={"PARNCP26CD": "category", "geometry": "string"},
        usecols=["PARNCP26CD", "geometry"],
    )
    parish_gdf.dropna(subset=["geometry", "PARNCP26CD"], inplace=True)

    parish_gdf = gpd.GeoDataFrame(
        parish_gdf,
        crs=EPSG_27700_CRS,
    )
    parish_gdf[_PARISH_GEOMETRY_COLUMN] = parish_gdf.geometry

    return parish_gdf


def _get_article_4_direction_gdf() -> gpd.GeoDataFrame:
    """Get Article 4 Direction GDF"""

    article_4_direction_df: pd.DataFrame = pd.read_csv(
        _ARTICLE_4_DIRECTION_FILE,
        date_format="%Y-%m-%d",
        parse_dates=["start-date", "end-date"],
        usecols=["end-date", "start-date", "geometry"],
        dtype={"geometry": "string"},
        keep_default_na=False,
    )
    article_4_direction_df.rename(
        columns={"start-date": _ARTICLE_4_START_DATE_COLUMN, "end-date": _ARTICLE_4_END_DATE_COLUMN}, inplace=True
    )
    article_4_direction_df.dropna(subset=["geometry"], inplace=True)

    article_4_direction_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        article_4_direction_df,
        geometry=gpd.GeoSeries.from_wkt(article_4_direction_df["geometry"], on_invalid="fix"),
        crs=EPSG_4326_CRS,
    ).to_crs(EPSG_27700_CRS)
    article_4_direction_gdf[_ARTICLE_4_DIRECTION_GEOMETRY_COLUMN] = article_4_direction_gdf.geometry

    return article_4_direction_gdf


def _plot_administrative_data(
    applications_gdf: gpd.GeoDataFrame, parishes_series: gpd.GeoSeries, administrative_geometry_column: str, title: str, label: str
) -> None:
    """Plot Environmental Data Against Applications"""
    if NO_PLOTS:
        return

    handles: list[Line2D] = []
    x_lim, y_lim = get_plot_lims(applications_gdf)
    x_dist: float = abs(x_lim[1] - x_lim[0])
    y_dist: float = abs(y_lim[1] - y_lim[0])
    aspect_ratio: float = y_dist / x_dist
    target_width: int = 1000
    target_height: int = int(target_width * aspect_ratio)
    cvs: ds.Canvas = ds.Canvas(
        plot_width=target_width, plot_height=target_height, x_range=(x_lim[0], x_lim[1]), y_range=(y_lim[0], y_lim[1])
    )

    administrative_geometry_series = gpd.GeoSeries(
        applications_gdf[administrative_geometry_column],
        crs=EPSG_27700_CRS,
    )
    administrative_geometry_series = administrative_geometry_series[administrative_geometry_series.geometry.notna()].copy()
    d_series = dgpd.from_geopandas(administrative_geometry_series, npartitions=os.cpu_count())
    d_simplified_series = d_series.geometry.simplify(tolerance=100, preserve_topology=False)
    simplified_series: GeoSeries = d_simplified_series.compute()
    simplified_administrative_geometry_df: pd.DataFrame = simplified_series.to_frame(name="geometry")

    print(simplified_administrative_geometry_df.head())
    parishes_agg: tf.Image = cvs.line(parishes_series.to_frame(name="geometry"), geometry="geometry")
    parishes_agg = tf.shade(parishes_agg, cmap=["#000000", "#000000"], alpha=100)
    applications_agg: tf.Image = cvs.points(applications_gdf, geometry="geometry")
    applications_agg = tf.shade(applications_agg, cmap=["#ff2c2c", "#ff2c2c"], alpha=200)

    administrative_agg_alpha: int = 200

    if administrative_geometry_column == _BROWNFIELD_GDF_GEOMETRY_COLUMN:
        administrative_agg: tf.Image = cvs.points(simplified_administrative_geometry_df, geometry="geometry")
    else:
        administrative_agg = cvs.polygons(simplified_administrative_geometry_df, geometry="geometry")
        administrative_agg_alpha = 20

    administrative_agg = tf.shade(administrative_agg, cmap=["#2196F3", "#2196F3"], alpha=administrative_agg_alpha)

    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="#2196F3",
            markerfacecolor="#2196F3",
            markeredgecolor="#2196F3",
            markersize=6,
            linestyle="None",
            alpha=0.5,
            label=label,
        )
    )

    images: tuple[tf.Image, ...] = (parishes_agg, applications_agg, administrative_agg)

    final_img = tf.stack(*tuple(tf.spread(i, px=1) for i in images))
    img_array = final_img.to_pil()

    fig_width: int = 10
    fig_height: float = fig_width * aspect_ratio
    _, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.set_aspect("equal")
    ax.set_facecolor("#e0e0e0")
    _ = ax.imshow(img_array, extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]))
    ax.set_axis_off()

    _ = ax.set_title(title)

    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="red",
            markerfacecolor="red",
            markeredgecolor="red",
            markersize=6,
            linestyle="None",
            alpha=0.5,
            label="Applications",
        )
    )

    ax.legend(handles=handles)

    plt.savefig(f"{title}.png", bbox_inches="tight")
    plt.show()


def _filter_by_start_end_date(applications_gdf: gpd.GeoDataFrame, start_date_column: str, end_date_column: str) -> pd.Series:
    decision_date = applications_gdf["Decision Date"]
    start_date = applications_gdf[start_date_column]
    end_date = applications_gdf[end_date_column]

    return decision_date.isna() | start_date.isna() | end_date.isna() | ((start_date <= decision_date) & (decision_date <= end_date))


def _join_brownfields_gdf(applications_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Get and Join Brownfields GDF to Applications GDF"""
    distance_col: Final[str] = "distance_m"
    brownfields_gdf: gpd.GeoDataFrame = _get_brownfields_gdf()
    applications_gdf = gpd.sjoin_nearest(
        applications_gdf,
        brownfields_gdf[[_BROWNFIELD_START_DATE_COLUMN, _BROWNFIELD_END_DATE_COLUMN, "geometry", _BROWNFIELD_GDF_GEOMETRY_COLUMN]],
        how="left",
        distance_col=distance_col,
        max_distance=10,
        rsuffix="brownfield_areas",
    )

    has_polygon_match = applications_gdf["index_brownfield_areas"].notna()
    applications_gdf[_BROWNFIELDS_CATEGORY_FIELD] = has_polygon_match & _filter_by_start_end_date(
        applications_gdf, _BROWNFIELD_START_DATE_COLUMN, _BROWNFIELD_END_DATE_COLUMN
    )
    brownfield_match = applications_gdf.groupby(_APPLICATIONS_GDF_ID)[_BROWNFIELDS_CATEGORY_FIELD].any()
    applications_gdf[_BROWNFIELDS_CATEGORY_FIELD] = applications_gdf[_APPLICATIONS_GDF_ID].map(brownfield_match).fillna(False).astype(int)

    print(applications_gdf[distance_col].describe())

    return applications_gdf.drop(
        columns=[_BROWNFIELD_START_DATE_COLUMN, _BROWNFIELD_END_DATE_COLUMN, distance_col, "index_brownfield_areas"]
    )


def _join_built_up_areas_gdf(applications_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Get and Join Built-Up Areas GDF to Applications GDF"""
    built_up_areas_gdf: gpd.GeoDataFrame = _get_built_up_areas_gdf()
    applications_gdf = gpd.sjoin(
        applications_gdf,
        built_up_areas_gdf[["geometry", _BUILT_UP_AREAS_GDF_GEOMETRY_COLUMN]],
        how="left",
        predicate="within",
        rsuffix="built_up_areas",
    )
    applications_gdf[_BUILT_UP_AREAS_CATEGORY_FIELD] = applications_gdf.groupby(level=0)["index_built_up_areas"].apply(is_within_area)

    return applications_gdf.drop(columns=["index_built_up_areas"])


def _join_special_areas_of_conservation_gdf(applications_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Get and Join Special Areas GDF to Applications GDF"""

    special_areas_of_conservation_gdf: gpd.GeoDataFrame = _get_special_areas_of_conservation_gdf()
    applications_gdf = gpd.sjoin(
        applications_gdf,
        special_areas_of_conservation_gdf[["geometry", _SPECIAL_AREAS_OF_CONSERVATION_GEOMETRY_COLUMN]],
        how="left",
        predicate="within",
        rsuffix="special_areas_of_conservation",
    )

    applications_gdf[_SPECIAL_AREAS_OF_CONSERVATION_CATEGORY_FIELD] = (
        applications_gdf["index_special_areas_of_conservation"].notna().astype(int)
    )

    return applications_gdf.drop(columns=["index_special_areas_of_conservation"])


def _join_parish_gdf(applications_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Get and Join Parish GDF to Applications GDF"""

    parish_gdf: gpd.GeoDataFrame = _get_parish_gdf()
    applications_gdf = gpd.sjoin(
        applications_gdf,
        parish_gdf[["PARNCP26CD", "geometry", _PARISH_GEOMETRY_COLUMN]],
        how="left",
        predicate="within",
        rsuffix="parish",
    )
    applications_gdf.drop(columns=["index_parish"], inplace=True)

    return applications_gdf.rename(columns={"PARNCP26CD": _PARISH_CATEGORY_FIELD})


def _join_article_4_directions_gdf(applications_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Get and Join article 4 Directions GDF to Applications GDF"""

    article_4_direction_gdf: gpd.GeoDataFrame = _get_article_4_direction_gdf()
    applications_gdf = (
        gpd.sjoin(
            applications_gdf,
            article_4_direction_gdf[
                ["geometry", _ARTICLE_4_START_DATE_COLUMN, _ARTICLE_4_END_DATE_COLUMN, _ARTICLE_4_DIRECTION_GEOMETRY_COLUMN]
            ],
            how="left",
            predicate="within",
            rsuffix="article_4_direction",
        )
        .sort_values("index_article_4_direction")
        .drop_duplicates(subset=_APPLICATIONS_GDF_ID, keep="first")
    )

    has_polygon_match = applications_gdf["index_article_4_direction"].notna()
    applications_gdf[_ARTICLE_4_DIRECTION_CATEGORY_FIELD] = has_polygon_match & _filter_by_start_end_date(
        applications_gdf, _ARTICLE_4_START_DATE_COLUMN, _ARTICLE_4_END_DATE_COLUMN
    )
    article_4_match = applications_gdf.groupby(_APPLICATIONS_GDF_ID)[_ARTICLE_4_DIRECTION_CATEGORY_FIELD].any()
    applications_gdf[_ARTICLE_4_DIRECTION_CATEGORY_FIELD] = (
        applications_gdf[_APPLICATIONS_GDF_ID].map(article_4_match).fillna(False).astype(int)
    )

    return applications_gdf.drop(columns=[_ARTICLE_4_START_DATE_COLUMN, _ARTICLE_4_END_DATE_COLUMN, "index_article_4_direction"])


def _plot_areas(applications_gdf: gpd.GeoDataFrame) -> None:
    """Plot Administrative Areas Against Applications"""
    parish_geometry_series: GeoSeries = gpd.GeoSeries(
        applications_gdf[_PARISH_GEOMETRY_COLUMN],
        crs=EPSG_27700_CRS,
    )

    d_parish_geometry_series = dgpd.from_geopandas(parish_geometry_series, npartitions=os.cpu_count())
    d_simplified_parish_geometry_series = d_parish_geometry_series.geometry.simplify(tolerance=10, preserve_topology=False)
    simplified_parish_geometry_series: GeoSeries = d_simplified_parish_geometry_series.compute()

    _plot_administrative_data(
        applications_gdf,
        simplified_parish_geometry_series,
        _BROWNFIELD_GDF_GEOMETRY_COLUMN,
        "Applications with Parishes and Brownfield Sites",
        "Brownfield Site",
    )

    _plot_administrative_data(
        applications_gdf,
        simplified_parish_geometry_series,
        _BUILT_UP_AREAS_GDF_GEOMETRY_COLUMN,
        "Applications with Parishes and Built Up Areas",
        "Built-Up Area",
    )
    _plot_administrative_data(
        applications_gdf,
        simplified_parish_geometry_series,
        _ARTICLE_4_DIRECTION_GEOMETRY_COLUMN,
        "Applications with Parishes and Article 4 Directions",
        "Article-4 Area",
    )


def preprocess_administrative_data(applications_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Preprocess and Append GeoSpatial Administrative Data"""

    applications_gdf_crs: CRS | None = applications_gdf.crs
    applications_gdf_wgs_27700 = applications_gdf.to_crs(EPSG_27700_CRS)
    # Add an id for subsequent deduplication.
    applications_gdf_wgs_27700[_APPLICATIONS_GDF_ID] = range(len(applications_gdf_wgs_27700))

    if not applications_gdf_crs:
        raise ValueError("Unknown applications CRS")

    applications_gdf_wgs_27700 = _join_brownfields_gdf(applications_gdf_wgs_27700)
    applications_gdf_wgs_27700 = _join_built_up_areas_gdf(applications_gdf_wgs_27700)
    applications_gdf_wgs_27700 = _join_special_areas_of_conservation_gdf(applications_gdf_wgs_27700)
    applications_gdf_wgs_27700 = _join_parish_gdf(applications_gdf_wgs_27700)
    applications_gdf_wgs_27700 = _join_article_4_directions_gdf(applications_gdf_wgs_27700)

    for category_field in _CATEGORY_FIELDS:
        print(f"{applications_gdf_wgs_27700.groupby(category_field).size()}\n")

    print(applications_gdf_wgs_27700.head())
    print(applications_gdf_wgs_27700.shape)

    _plot_areas(applications_gdf_wgs_27700)
    applications_gdf_wgs_27700.drop(
        columns=[
            _APPLICATIONS_GDF_ID,
            _BROWNFIELD_GDF_GEOMETRY_COLUMN,
            _SPECIAL_AREAS_OF_CONSERVATION_GEOMETRY_COLUMN,
            _PARISH_GEOMETRY_COLUMN,
            _ARTICLE_4_DIRECTION_GEOMETRY_COLUMN,
            _BUILT_UP_AREAS_GDF_GEOMETRY_COLUMN,
        ],
        inplace=True,
    )

    return applications_gdf_wgs_27700.to_crs(applications_gdf_crs)
