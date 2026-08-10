"""Preprocess the Raw Planning Application File for Feature Development"""

from pathlib import Path
from typing import Any, Final

from pandas import DataFrame, read_csv

ADDRESS_COLUMN: Final[str] = "address-text"
COLUMN_DTYPES: Final[dict[str, str]] = {
    "entity": "string",
    "organisation-entity": "category",
    "point": "string",  # Treat as a string for further preprocessing
    ADDRESS_COLUMN: "string",
    "description": "string",
    "development-classification": "category",
    "planning-application-status": "category",
    "planning-decision": "category",  # Target
    "planning-decision-type": "category",
}
DATE_COLUMNS: Final[list[str]] = ["entry-date", "start-date", "decision-date"]
EXTRANEOUS_COLUMNS: Final[list[str]] = [
    "dataset",
    "end-date",
    "geojson",
    "geometry",
    "name",
    "prefix",
    "reference",
    "typology",
    "documentation-url",
    "ground-area",
    "notes",
    "organisation",
    "planning-application-type",
    "uprn",
]
PLANNING_DATA_FILE = Path(__file__).parent / "data" / "planning-application_20250917.csv"
POSTCODE_PATTERN: Final[str] = r"\b([A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})\b"


def _disaggregate_date_field(data: DataFrame, field: str) -> None:
    """Disaggregate date field into year and quarter category fields."""
    data["year-of-transfer"] = data[field].dt.year.astype("category")
    data["quarter-of-transfer"] = data[field].dt.quarter.astype("category")


def _remove_extraneous_data(data: DataFrame) -> DataFrame:
    """Remove extraneous columns and rows from data frame."""

    def _is_valid_text_field(value: object) -> bool:
        return isinstance(value, str) and bool(value.strip())

    data.drop(columns=EXTRANEOUS_COLUMNS, inplace=True)
    df = data[data[ADDRESS_COLUMN].apply(_is_valid_text_field)]

    return df[df["description"].apply(_is_valid_text_field)]


def _disaggregate_address_field(data: DataFrame) -> None:
    """Disaggregate address field into component fields"""
    # Extract the postcode from the address if present and remove from address
    data["postcode"] = data[ADDRESS_COLUMN].str.extract(POSTCODE_PATTERN, expand=False)
    data[ADDRESS_COLUMN] = (
        data[ADDRESS_COLUMN]
        .str.replace(POSTCODE_PATTERN, "", regex=True)
        .str.replace(r"\s*,\s*$", "", regex=True)
        .str.strip()
    )
    # Extract the Outward Code from Postcode
    data["outward-code"] = data["postcode"].str.split(n=1).str[0]
    data["outward-code"] = data["outward-code"].astype("category")


def preprocess_planning_data() -> DataFrame:
    """Preprocess the Raw Planning Application File for Feature Development"""
    preprocessed_data_df = read_csv(
        PLANNING_DATA_FILE,
        dtype=COLUMN_DTYPES,
        date_format="%Y-%m-%d",
        parse_dates=DATE_COLUMNS,
        keep_default_na=False,
    )

    for date_column in DATE_COLUMNS:
        _disaggregate_date_field(preprocessed_data_df, date_column)

    print(f"Raw shape: {preprocessed_data_df.shape}")
    preprocessed_data_df = _remove_extraneous_data(preprocessed_data_df)
    print(f"Shape without extraneous data: {preprocessed_data_df.shape}")

    preprocessed_data_df[ADDRESS_COLUMN] = preprocessed_data_df[ADDRESS_COLUMN].str.upper()
    preprocessed_data_df["description"] = preprocessed_data_df["description"].str.upper()

    _disaggregate_address_field(preprocessed_data_df)
    print(f"Observations with postcode: {preprocessed_data_df['postcode'].notna().sum()}")

    return preprocessed_data_df
