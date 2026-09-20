"""sBERT Transformer Wrapper for Scikit Learn Pipelines"""

from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class SentenceTransformerWrapper(BaseEstimator, TransformerMixin):
    """Pipeline Transformer for Applying sBERT encodings to target columns"""

    model_name: str
    model: SentenceTransformer | None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.model_name = model_name

    def fit(self, x, y=None):
        self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, x):
        if self.model is not None:
            return self.model.encode(x, normalize_embeddings=True, batch_size=64, device="cuda")
        else:
            raise ValueError("SentenceTransformer model not supplied")
