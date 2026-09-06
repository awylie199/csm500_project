"""Analyse and Evaluate Correlation between Independent Explanatory Fields"""

import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def correlation_analysis(x_train: gpd.GeoDataFrame) -> None:
    """Spearman's Rank Correlation for improved handling of ordinal categories and missing values"""
    # Create pairwise interaction features to capture potential explanatory value from the combination of 2 fields
    pf_correlation_analysis_df = x_train.copy(deep=True).drop(columns=["Description", "Street"])
    pf_correlation_analysis_squared_df = pf_correlation_analysis_df.copy(deep=True)
    polynomial_features_transformer: PolynomialFeatures = PolynomialFeatures(include_bias=False, interaction_only=True)
    polynomial_features_squared_transformer: PolynomialFeatures = PolynomialFeatures(include_bias=False)

    pf_df = pd.DataFrame(
        polynomial_features_transformer.fit_transform(pf_correlation_analysis_df),
        columns=polynomial_features_transformer.get_feature_names_out(),
        index=x_train.index,
    )
    pf_squared_df = pd.DataFrame(
        polynomial_features_squared_transformer.fit_transform(pf_correlation_analysis_squared_df),
        columns=polynomial_features_squared_transformer.get_feature_names_out(),
        index=x_train.index,
    )
