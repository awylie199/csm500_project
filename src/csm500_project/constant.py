"""Shared constants."""

from pathlib import Path
from typing import Final

INFERRED_TARGET_FIELD: Final[str] = "Inferred Planning Decision"
NO_PLOTS: Final[bool] = True
USE_UPRN_CACHE: Final[bool] = False
EPSG_27700_CRS: Final[str] = "EPSG:27700"
EPSG_4326_CRS: Final[str] = "EPSG:4326"

RAW_TRAIN_DATASET_X_PATH: Final[Path] = Path(__file__).parent / "cache" / "raw_train_dataset_x.parquet"
RAW_TRAIN_DATASET_Y_PATH: Final[Path] = Path(__file__).parent / "cache" / "raw_train_dataset_y.parquet"
RAW_TEST_DATASET_X_PATH: Final[Path] = Path(__file__).parent / "cache" / "raw_test_dataset_x.parquet"
RAW_TEST_DATASET_Y_PATH: Final[Path] = Path(__file__).parent / "cache" / "raw_test_dataset_y.parquet"

FINAL_TRAIN_PRIMARY_DATASET_PATH: Final[Path] = Path(__file__).parent / "cache" / "final_train_primary_dataset.parquet"
FINAL_TRAIN_PHYSICAL_DATASET_PATH: Final[Path] = Path(__file__).parent / "cache" / "final_train_physical_dataset.parquet"

BAGGING_CLASSIFIER_KNEIGHBOURS_DUMP: Final[str] = "bagging_classifier_kneighbours.joblib"
RANDOM_FOREST_FILE_DUMP: Final[str] = "random_forest.joblib"
ADABOOST_DECISION_TREE_DUMP: Final[str] = "adaboost_decision_tree.joblib"
