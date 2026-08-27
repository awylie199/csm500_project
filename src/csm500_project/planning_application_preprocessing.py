"""Preprocess the Raw Planning Application File for Feature Development"""

from pathlib import Path
from typing import Final

import pandas as pd

from .constant import INFERRED_TARGET_FIELD
from .infer_target_field import infer_target_field
from .util import is_not_empty

POINT_COLUMN: Final[str] = "point"
COLUMN_DTYPES: Final[dict[str, str]] = {
    "organisation-entity": "category",
    POINT_COLUMN: "string",  # Treat initially as a string for later parsing as a Point type
    "description": "string",
    "development-classification": "category",
    "planning-decision": "category",
    "planning-decision-type": "category",
}
DATE_COLUMNS: Final[list[str]] = ["entry-date", "decision-date"]
PLANNING_DATA_FILE: Final[Path] = Path(__file__).parent / "data" / "planning-application_20250917.csv"
COLUMNS: Final[list[str]] = [
    "organisation-entity",
    POINT_COLUMN,
    "description",
    "development-classification",
    "planning-decision",
    "planning-decision-type",
    *DATE_COLUMNS,
]
YEAR_OF_TRANSFER: Final[str] = "year-of-transfer"
QUARTER_OF_TRANSFER: Final[str] = "quarter-of-transfer"


def _disaggregate_date_field(data: pd.DataFrame, field: str) -> None:
    """Disaggregate date field into year and quarter category fields."""
    data[f"{field}-{YEAR_OF_TRANSFER}"] = data[field].dt.year.astype("category")
    data[f"{field}-{QUARTER_OF_TRANSFER}"] = data[field].dt.quarter.astype("category")


def _remove_extraneous_data(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove extraneous columns and rows from data frame."""

    # Remove redundant columns, descriptions and points
    data.dropna(inplace=True)

    return data[data[columns].map(is_not_empty).all(axis=1)]


def preprocess_planning_data() -> pd.DataFrame:
    """Preprocess the Raw Planning Application File for Feature Development"""
    preprocessed_data_df = pd.read_csv(
        PLANNING_DATA_FILE,
        dtype=COLUMN_DTYPES,
        usecols=COLUMNS,
        date_format="%Y-%m-%d",
        parse_dates=DATE_COLUMNS,
        keep_default_na=False,
    )

    print(f"Raw shape: {preprocessed_data_df.shape}")
    print(f"Raw Null Values:\n {preprocessed_data_df.isna().sum()}")

    for date_column in DATE_COLUMNS:
        _disaggregate_date_field(preprocessed_data_df, date_column)

    preprocessed_data_df.drop(columns=["entry-date"], inplace=True)

    preprocessed_data_df["description"] = preprocessed_data_df["description"].str.upper()

    print(f"{preprocessed_data_df.groupby('planning-decision').size()}\n")
    print(f"{preprocessed_data_df.groupby('planning-decision-type').size()}\n")

    preprocessed_data_df[INFERRED_TARGET_FIELD] = preprocessed_data_df.apply(infer_target_field, axis=1)
    print(f"{preprocessed_data_df.groupby(INFERRED_TARGET_FIELD).size()}\n")
    preprocessed_data_df.drop(columns=["planning-decision", "planning-decision-type"], inplace=True)
    # Remove unlabelled data that we cannot infer, or where no decision was made (e.g. withdrawn)
    preprocessed_data_df = preprocessed_data_df[
        ~preprocessed_data_df[INFERRED_TARGET_FIELD].isin(["Prior not required - prior approval not required", "Withdrawn"])
    ]

    preprocessed_data_df = _remove_extraneous_data(preprocessed_data_df, ["description", POINT_COLUMN, INFERRED_TARGET_FIELD])
    print(f"{preprocessed_data_df.groupby(INFERRED_TARGET_FIELD).size()}\n")

    preprocessed_data_df.rename(
        columns={c: c.replace("-", " ").title() for c in list(preprocessed_data_df.columns.values)},
        inplace=True,
    )

    return preprocessed_data_df
