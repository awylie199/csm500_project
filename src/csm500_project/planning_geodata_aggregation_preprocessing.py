from pathlib import Path
from typing import Final

import geopandas as gpd
from pandas import DataFrame

LSOA_GPKG_FILE: Final[Path] = (
    Path(__file__).parent
    / "data"
    / "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5_-6970154227154374572.gpkg"
)


def preprocess_aggregate_geodata(df: DataFrame) -> gpd.GeoDataFrame:
    """Aggregate geodata data sources using LSOA boundary polygons"""

    gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        df.copy(), geometry=gpd.GeoSeries.from_wkt(df["point"]), crs="EPSG:4326"
    )

    lsoa: gpd.GeoDataFrame = gpd.read_file(LSOA_GPKG_FILE)
    lsoa = lsoa.to_crs(gdf.crs)

    print(f"GeoDataFrame CRS: {gdf.crs}")
    print(f"LSOA CRS: {lsoa.crs}")

    gdf = gpd.sjoin(gdf, lsoa[["LSOA21CD", "LSOA21NM", "geometry"]], how="left", predicate="within")
    return gdf


#
#     # Plot POINT() onto basemap of England
# https://geopandas.org/en/stable/gallery/plotting_basemap_background.html
#
#
