"""Feature Selection for the Final Dataset for Downstream Model Development"""

from typing import Final

import pandas as pd
from sklearn.metrics import silhouette_score

from .constant import INFERRED_TARGET_FIELD

_RANDOM_STATE: Final[int] = 20260909


def _compare_text_representation(text_transforms_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> list[tuple[str, float | int]]:
    """
    Use silhouette score to compare text representations

    Infer: - if sBERT is higher, application result depends on underlying semantic structure etc.
           - if TF-IDF or CV is higher, Application result depends more on distinct keywords that embeddings smooth over
           - if similar, classes are similarly clustered and less expensive method is probably preferred
    """
    scores: list[tuple[str, float | int]] = []

    for i, text_transform in enumerate(text_transforms_dfs):
        y: pd.Series = text_transform[INFERRED_TARGET_FIELD]
        x: pd.DataFrame = text_transform.drop(columns=[INFERRED_TARGET_FIELD])

        score: float | int = silhouette_score(x, y, metric="cosine", sample_size=5000, random_state=_RANDOM_STATE)
        scores.append(("Count Vectorizer" if i == 0 else "TF-IDF" if i == 1 else "sBERT", score))

    return scores


def feature_selection(
    x_train: pd.DataFrame, y_train: pd.DataFrame, text_transforms_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
) -> None:
    """Select features for final dataset ready for downstream model development"""

    for method, score in _compare_text_representation(text_transforms_dfs):
        print(f"Text Representation {method} Silhouette Score: {score}")
