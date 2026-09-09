"""Discrete Data Analysis"""

from collections.abc import Callable
from typing import Final

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from statsmodels.graphics.mosaicplot import mosaic

from .constant import INFERRED_TARGET_FIELD

_ORDINAL_CATEGORIES: Final[tuple[str, ...]] = (
    "Entry Date Year Of Transfer",
    "Entry Date Quarter Of Transfer",
    "Decision Date Year Of Transfer",
    "Decision Date Quarter Of Transfer",
    "IMD Rank",
    "IMD Decile",
    "Listing Grade",
)
_NON_ORDINAL_CATEGORIES: Final[tuple[str, ...]] = (
    "Organisation Entity",
    "Development Classification",
    "LSOA Code",
    "Council Code",
    "Geohazards Collapsible Deposits Susceptibility",
    "Geohazards Compressible Ground Susceptibility",
    "Geohazards Landslides Susceptibility",
    "Geohazards Running Sand Susceptibility",
    "Geohazards Shrink Swell Susceptibility",
    "Geohazards Soluble Rocks Susceptibility",
    "In Conservation Area",
    "In Ancient Woodlands",
    "In Green Belt",
    "In Flood Zone",
    "Property Type",
    "Is Domestic",
    "Near Brownfield Site",
    "In Built Up Area",
    "In Special Area of Conservation",
    "Parish Code",
    "In Article 4 Area",
)
_DROPPED_CATEGORIES: Final[tuple[str, ...]] = (
    "Entry Date Year Of Transfer",
    "Entry Date Quarter Of Transfer",
    "Geohazards Compressible Ground Susceptibility",
    "Geohazards Landslides Susceptibility",
    "In Ancient Woodlands",
    "In Green Belt",
    "In Special Area of Conservation",
)
_BINNED_CATEGORIES: Final[tuple[str, ...]] = ("IMD Decile",)
_X_LABEL_ENCODED_CATEGORIES: Final[tuple[str, ...]] = (
    "Geohazards Collapsible Deposits Susceptibility",
    "Geohazards Running Sand Susceptibility",
    "Geohazards Shrink Swell Susceptibility",
    "Geohazards Soluble Rocks Susceptibility",
)
_HIGH_CARDINALITY_ENCODED_CATEGORIES: Final[tuple[str, ...]] = ("LSOA Code", "Development Classification", "Property Type")
_LOW_CARDINALITY_ENCODED_CATEGORIES: Final[tuple[str, ...]] = ("Organisation Entity", "Council Code", "Listing Grade", "Parish Code")
_DUMMY_ENCODED_CATEGORIES: Final[tuple[str, ...]] = (
    "Organisation Entity_26",
    "Organisation Entity_382",
    "Organisation Entity_90",
    "Council Code_E09000003",
    "Council Code_E09000007",
    "Council Code_E09000014",
    "Council Code_E09000019",
    "Council Code_E09000033",
    "Council Code_E10000032",
    "Listing Grade_1",
    "Listing Grade_2",
    "Listing Grade_3",
    "Parish Code_E04009848",
    "Parish Code_E43000147",
    "Parish Code_E43000150",
    "Parish Code_E43000191",
    "Parish Code_E43000193",
    "Parish Code_E43000197",
    "Parish Code_E43000204",
)
FINAL_CATEGORIES: Final[tuple[str, ...]] = (
    *_DUMMY_ENCODED_CATEGORIES,
    *[
        cat
        for cat in [*_ORDINAL_CATEGORIES, *_NON_ORDINAL_CATEGORIES, *_BINNED_CATEGORIES]
        if cat not in [*_LOW_CARDINALITY_ENCODED_CATEGORIES, *_DROPPED_CATEGORIES]
    ],
)
_MOSAIC_PROPERTIES: Final[dict[tuple[str, str], dict[str, str]]] = {
    ("0", "False"): {"facecolor": "lightpink", "hatch": "/", "edgecolor": "lightpink"},
    ("1", "False"): {"facecolor": "steelblue"},
    ("2", "False"): {"facecolor": "palegoldenrod"},
    ("0", "True"): {"facecolor": "palevioletred"},
    ("1", "True"): {"facecolor": "royalblue", "hatch": "/"},
    ("2", "True"): {"facecolor": "sandybrown"},
}


def _get_mosaic_labeliser(label: str) -> Callable[..., str]:
    formatted_label: str = label.split("_")[-1]

    return lambda k: {  # pyright: ignore[reportUnknownVariableType, reportUnknownLambdaType]
        ("0", "False"): f"Accept \n {formatted_label}: False",
        ("1", "False"): f"Refuse \n {formatted_label}: False",
        ("2", "False"): f"Split \n {formatted_label}: False",
        ("0", "True"): f"Accept \n {formatted_label}: True",
        ("1", "True"): f"Refuse \n {formatted_label}: True",
        ("2", "True"): f"Split \n {formatted_label}: True",
    }[k]


def _plot_mosaic_contingency(encoded_applications_df: pd.DataFrame, x_field: str, title: str) -> None:
    """Plot a Mosaic Showing Relationship Between Target and the X Field"""
    _, ax = plt.subplots(figsize=(10, 6))
    _ = mosaic(
        encoded_applications_df,
        [
            INFERRED_TARGET_FIELD,
            x_field,
        ],
        properties=_MOSAIC_PROPERTIES,
        axes_label=True,
        labelizer=_get_mosaic_labeliser(x_field),
        gap=0.02,
        ax=ax,
    )

    _ = ax.set_title(title.replace("_", " "), pad=40)

    plt.tight_layout()

    for text in ax.texts:
        text.set_fontsize(8)
        text.set_rotation(45)
        text.set_horizontalalignment("center")

    plt.show()


def discrete_data_analysis(x_train: gpd.GeoDataFrame, y_train: gpd.GeoDataFrame) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Analyse the discrete data within the aggregated applications dataset"""

    for category in [*_NON_ORDINAL_CATEGORIES, *_ORDINAL_CATEGORIES]:
        if category not in _BINNED_CATEGORIES:
            print(f"{x_train.groupby(category).size()}\n")

    print(f"{y_train.groupby(INFERRED_TARGET_FIELD).size()}\n")

    # No variance
    x_train.drop(
        columns=list(_DROPPED_CATEGORIES),
        inplace=True,
    )

    # Map labels to integers on target field - may affect Accuracy metric
    y_train[INFERRED_TARGET_FIELD] = LabelEncoder().fit_transform(y_train[INFERRED_TARGET_FIELD])
    print(f"{y_train.groupby(INFERRED_TARGET_FIELD).size()}\n")

    for category in _X_LABEL_ENCODED_CATEGORIES:
        x_train[category] = LabelEncoder().fit_transform(x_train[category])

    # Apply high amount of smoothing due to high risk of data-leakage introduced by TargetEncoder (by its use of target field)
    target_encoder: TargetEncoder = TargetEncoder(smoothing=10)

    for category in _HIGH_CARDINALITY_ENCODED_CATEGORIES:
        x_train[category] = target_encoder.fit_transform(x_train[category], y_train)  # pyright: ignore[reportArgumentType]

    # Drop First to avoid perfect multicollinearity in linear models
    dummy_encoded_x_train: pd.DataFrame = pd.get_dummies(x_train, columns=[*_LOW_CARDINALITY_ENCODED_CATEGORIES], drop_first=True)

    print(y_train.dtypes)
    print(dummy_encoded_x_train.dtypes)

    contingency_df: pd.DataFrame = dummy_encoded_x_train.copy(deep=True)
    contingency_df[INFERRED_TARGET_FIELD] = y_train[INFERRED_TARGET_FIELD]

    # Observe contingency between target and dummy encoded categories
    for category in _DUMMY_ENCODED_CATEGORIES:
        _plot_mosaic_contingency(contingency_df, category, f"Contingency Between Application Result and {category}")

    return dummy_encoded_x_train, y_train
