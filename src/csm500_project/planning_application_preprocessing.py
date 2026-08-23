"""Preprocess the Raw Planning Application File for Feature Development"""

from pathlib import Path
from typing import Final

import pandas as pd

from .util import is_not_empty

ADDRESS_COLUMN: Final[str] = "address-text"
POINT_COLUMN: Final[str] = "point"
COLUMN_DTYPES: Final[dict[str, str]] = {
    "entity": "string",
    "organisation-entity": "category",
    POINT_COLUMN: "string",  # Treat initially as a string for later parsing as a Point type
    ADDRESS_COLUMN: "string",
    "description": "string",
    "development-classification": "category",
    "planning-application-status": "category",
    "planning-decision": "category",  # Target
    "planning-decision-type": "category",
}
DATE_COLUMNS: Final[list[str]] = ["entry-date", "decision-date"]
PLANNING_DATA_FILE: Final[Path] = Path(__file__).parent / "data" / "planning-application_20250917.csv"
COLUMNS: Final[list[str]] = [
    "entity",
    "organisation-entity",
    POINT_COLUMN,
    ADDRESS_COLUMN,
    "description",
    "development-classification",
    "planning-application-status",
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

    data.drop(columns={field}, inplace=True)


def _remove_extraneous_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove extraneous columns and rows from data frame."""

    # Remove redundant columns, observations without addresses, descriptions and points
    data.dropna(inplace=True)
    df = data[data[ADDRESS_COLUMN].apply(is_not_empty)]
    df = df[df[POINT_COLUMN].apply(is_not_empty)]

    return df[df["description"].apply(is_not_empty)]


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
    preprocessed_data_df = _remove_extraneous_data(preprocessed_data_df)
    print(f"Shape without extraneous data: {preprocessed_data_df.shape}")

    for date_column in DATE_COLUMNS:
        _disaggregate_date_field(preprocessed_data_df, date_column)

    preprocessed_data_df[ADDRESS_COLUMN] = preprocessed_data_df[ADDRESS_COLUMN].str.upper()
    preprocessed_data_df["description"] = preprocessed_data_df["description"].str.upper()

    preprocessed_data_df.rename(
        columns={c: c.replace("-", " ").title() for c in list(preprocessed_data_df.columns.values)},
        inplace=True,
    )

    return preprocessed_data_df
