"""Mutual Info Scores Wrapper for Scikit Learn Pipelines to Drop Less than 1%"""

from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_is_fitted, validate_data

_RANDOM_STATE: Final[int] = 20260916


class MutualInfoScoresTransformer(BaseEstimator, TransformerMixin):
    """Pipeline Transformer for Mutual Scores and Dropping Features Below Threshold"""

    scores_: NDArray[Any]  # pyright: ignore[reportUninitializedInstanceVariable]
    input_features_: NDArray[Any]  # pyright: ignore[reportUninitializedInstanceVariable]
    feature_names_in_: NDArray[Any]  # pyright: ignore[reportUninitializedInstanceVariable]
    support_: NDArray[np.bool_]  # pyright: ignore[reportUninitializedInstanceVariable]
    threshold: float = 0.01

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def fit(self, x, y):
        x, y = validate_data(
            self,
            x,
            y,
            reset=True,
        )

        self.scores_: NDArray[np.float64] = mutual_info_classif(
            x,
            y,
            random_state=_RANDOM_STATE,
        )

        self.support_ = self.scores_ >= self.threshold

        return self

    def transform(self, x) -> pd.DataFrame:
        check_is_fitted(self, ["scores_", "support_"])

        original_index = x.index if isinstance(x, pd.DataFrame) else None

        x_validated = validate_data(
            self,
            x,  # pyright: ignore[reportArgumentType]
            reset=False,
        )

        feature_names = self.get_feature_names_out()

        return pd.DataFrame(
            x_validated[:, self.support_],  # pyright: ignore[reportArgumentType, reportCallIssue]
            columns=feature_names,
            index=original_index,
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["scores_", "support_"])

        if input_features is None:
            input_features = self.feature_names_in_

        input_features = np.asarray(input_features, dtype=object)

        return input_features[self.support_]
