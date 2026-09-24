"""Pipeline Transformer for Examining Correlation and Drop all Multicollinear Columns"""

from typing import Any

import numpy as np
import pandas as pd
from numpy import dtype, ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class MulticollinearityDropTransformer(BaseEstimator, TransformerMixin):
    """Pipeline Transformer for Dropping Multicollinear Columns"""

    feature_names_in_: ndarray[tuple[int], dtype[Any]]  # pyright: ignore[reportUninitializedInstanceVariable]
    threshold: float = 0.95
    drop_columns: list[str]  # pyright: ignore[reportUninitializedInstanceVariable]

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, x, y=None):
        x_df: pd.DataFrame = pd.DataFrame(x)
        self.feature_names_in_ = x_df.columns.to_numpy()
        x_correlation_df: pd.DataFrame = x_df.corr().abs()
        multicollinear_df: pd.DataFrame = x_correlation_df.where(np.triu(np.ones(x_correlation_df.shape), k=1).astype(bool))
        self.drop_columns = [column for column in multicollinear_df.columns if any(multicollinear_df[column] > self.threshold)]

        return self

    def transform(self, x) -> pd.DataFrame:
        check_is_fitted(self, ["feature_names_in_", "drop_columns"])
        return pd.DataFrame(x).drop(columns=self.drop_columns)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["feature_names_in_", "drop_columns"])

        if input_features is None:
            input_features = self.feature_names_in_
        else:
            input_features = np.asarray(input_features)

        return np.asarray([feature for feature in input_features if feature not in self.drop_columns])
