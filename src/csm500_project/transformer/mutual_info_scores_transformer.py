"""Mutual Info Scores Wrapper for Scikit Learn Pipelines to Drop Less than 1%"""

from typing import Any, Final

from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin, mutual_info_classif

_RANDOM_STATE: Final[int] = 20260916


class MutualInfoScoresTransformer(BaseEstimator, SelectorMixin):
    """Pipeline Transformer for Mutual Scores and Dropping Features Below Threshold"""

    scores: NDArray[Any] | None
    threshold: float = 0.01

    def __init__(self, threshold: float = 0.01):
        self.scores = None
        self.threshold = threshold

    def fit(self, x, y):
        self.scores = mutual_info_classif(
            x,
            y,
            random_state=_RANDOM_STATE,
        )
        return self

    def _get_support_mask(self):
        return self.scores >= self.threshold
