"""Discrete Data Analysis"""

from typing import Final

import geopandas as gpd

from .constant import INFERRED_TARGET_FIELD

ORDINAL_CATEGORIES: Final[tuple[str, ...]] = (
    "Entry Date Year Of Transfer",
    "Entry Date Quarter Of Transfer",
    "Decision Date Year Of Transfer",
    "Decision Date Quarter Of Transfer",
    "IMD Rank",
)
NON_ORDINAL_CATEGORIES: Final[tuple[str, ...]] = (
    "Organisation Entity",
    "Development Classification",
    "IMD Decile",
    "In Conservation Area",
)


def discrete_data_analysis(x_train: gpd.GeoDataFrame, y_train: gpd.GeoDataFrame) -> None:
    """Analyse the discrete data within the aggregated applications dataset"""

    for category in NON_ORDINAL_CATEGORIES:
        # IMD Decile is already binned category
        if category != "IMD Decile":
            print(f"{x_train.groupby(category).size()}\n")

    print(f"{y_train.groupby(INFERRED_TARGET_FIELD).size()}\n")

    for ordinal_category in ORDINAL_CATEGORIES:
        ordinal_category_len: int = len(x_train[ordinal_category].unique())
        print(f"Unique {ordinal_category} ({ordinal_category_len} of {len(x_train[ordinal_category])})\n")
