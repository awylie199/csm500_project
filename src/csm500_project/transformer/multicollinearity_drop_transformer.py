"""Pipeline Transformer for Examining Correlation and Drop all Multicollinear Columns"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MulticollinearityDropTransformer(BaseEstimator, TransformerMixin):
    """Pipeline Transformer for Dropping Multicollinear Columns"""

    threshold: float = 0.95
    drop_columns: list[str]

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.drop_columns = []

    def fit(self, x, y=None):
        x_df: pd.DataFrame = pd.DataFrame(x)
        x_correlation_df: pd.DataFrame = x_df.corr().abs()
        multicollinear_df: pd.DataFrame = x_correlation_df.where(np.triu(np.ones(x_correlation_df.shape), k=1).astype(bool))
        self.drop_columns = [column for column in multicollinear_df.columns if any(multicollinear_df[column] > self.threshold)]

        return self

    def transform(self, x):
        return pd.DataFrame(x).drop(columns=self.drop_columns).values
