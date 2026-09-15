"""Shared constants."""

from pathlib import Path
from typing import Final

INFERRED_TARGET_FIELD: Final[str] = "Inferred Planning Decision"
NO_PLOTS: Final[bool] = True
USE_UPRN_CACHE: Final[bool] = False
EPSG_27700_CRS: Final[str] = "EPSG:27700"
EPSG_4326_CRS: Final[str] = "EPSG:4326"

PRIMARY_DATASET_PATH: Final[Path] = Path(__file__).parent / "cache" / "primary_dataset.parquet"
PHYSICAL_DATASET_PATH: Final[Path] = Path(__file__).parent / "cache" / "physical_dataset.parquet"
