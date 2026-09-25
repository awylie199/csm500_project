"""sBERT Transformer Wrapper for Scikit Learn Pipelines"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class SentenceTransformerWrapper(BaseEstimator, TransformerMixin):
    """Pipeline Transformer for sBERT Embeddings"""

    model_name: str
    model_: SentenceTransformer  # pyright: ignore[reportUninitializedInstanceVariable]

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    def fit(self, x, y=None):
        self.model_ = SentenceTransformer(self.model_name)
        return self

    def transform(self, x):
        check_is_fitted(self, "model_")

        embeddings = self.model_.encode(
            x.iloc[:, 0].astype(str).tolist(),
            normalize_embeddings=True,
            batch_size=8,
            device="cuda",
        )

        result = pd.DataFrame(
            embeddings,
            columns=[f"sbert_{i}" for i in range(embeddings.shape[1])],
            index=x.index,
        )

        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "model_")
        n_embeddings: int | None = self.model_.get_embedding_dimension()

        if n_embeddings is None:
            raise ValueError("no sbert embedding dimensions")

        return np.array([f"sbert_{i}" for i in range(n_embeddings)])
