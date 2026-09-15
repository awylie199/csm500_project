"""Feature Selection for the Final Dataset for Downstream Model Development"""

from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

from .constant import INFERRED_TARGET_FIELD

_RANDOM_STATE: Final[int] = 20260909


def _compare_text_representation(
    text_transforms_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> list[tuple[str, float, float]]:
    """
    Utilise models to compare text representation.
    Employ Logistic regression for evaluating linear relationships in sparse datasets of CV and TFIDF
    Employ LGBM classifier for non-linear dense sBERT embeddings
    """
    cv_strategy: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=_RANDOM_STATE)
    scores: list[tuple[str, float, float]] = []

    for i, text_df in enumerate(text_transforms_dfs):
        y = text_df[INFERRED_TARGET_FIELD]
        x = text_df.drop(columns=[INFERRED_TARGET_FIELD])

        if i == 2:
            text_df_cross_val_score = cross_validate(
                LGBMClassifier(class_weight="balanced", n_estimators=100, learning_rate=0.05, n_jobs=-1, random_state=_RANDOM_STATE),
                x,
                y,
                cv=cv_strategy,
                scoring="f1_macro",
                return_train_score=True,
            )
        else:
            text_df_cross_val_score = cross_validate(
                LogisticRegression(class_weight="balanced", max_iter=500),
                x,
                y,
                cv=cv_strategy,
                scoring="f1_macro",
                return_train_score=True,
            )

        scores.append(
            (
                "CV" if i == 0 else "TFIDF" if i == 1 else "sBERT",
                np.mean(text_df_cross_val_score["train_score"]),
                np.mean(text_df_cross_val_score["test_score"]),
            )
        )

    return scores


def _get_continuous_data_scaler(x_train: pd.DataFrame) -> ColumnTransformer:
    """Pass continuous numeric data through a standard scaler"""
    continuous_numeric_columns: list[str] = [c for c in x_train.columns if "Floor Area (m^2)" in c]

    return ColumnTransformer(
        transformers=[("standard_scaler_transformer", StandardScaler(), continuous_numeric_columns)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def _variance_threshold_selection(
    x_train: pd.DataFrame,
    columns: pd.Index[str],
) -> pd.DataFrame:
    """
    Unsupervised Feature selection by Variance Threshold
    Embeddings are not included as the high dimensionality will obscure other features and exhibit low variance across extracted features
    """

    variance_threshold_transformer: VarianceThreshold = VarianceThreshold(threshold=0.01)
    variance_threshold_selection_df: pd.DataFrame = pd.DataFrame(
        variance_threshold_transformer.fit_transform(x_train),
        columns=variance_threshold_transformer.get_feature_names_out(input_features=x_train.columns),
        index=x_train.index,
    )

    print(f"\nVariance Threshold Features:\n{columns[variance_threshold_transformer.get_support()]}")  # pyright: ignore[reportCallIssue, reportArgumentType]

    return pd.DataFrame(variance_threshold_selection_df)


def _selectkbest_selection(x_train: pd.DataFrame, columns: pd.Index[str], y_train: pd.DataFrame) -> pd.DataFrame:
    """Supervised Feature Select by Select K Best with Mutual Info Classification"""

    select_kbest_transformer: SelectKBest = SelectKBest(mutual_info_classif)
    select_kbest_selection_df: pd.DataFrame = pd.DataFrame(
        select_kbest_transformer.fit_transform(x_train, y_train.values.ravel()),
        columns=select_kbest_transformer.get_feature_names_out(input_features=x_train.columns),
        index=x_train.index,
    )

    print(f"\nSelect K Best Features:\n{columns[select_kbest_transformer.get_support()]}")  # pyright: ignore[reportCallIssue, reportArgumentType]

    return select_kbest_selection_df


def _log_regression_selection(x_train: pd.DataFrame, columns: pd.Index[str], y_train: pd.DataFrame) -> pd.DataFrame:
    """Use log regression to select features"""
    log_regression_selection_transformer: SelectFromModel = SelectFromModel(
        estimator=LogisticRegression(class_weight="balanced", max_iter=500), threshold=None
    )
    log_regression_selection_df: pd.DataFrame = pd.DataFrame(
        log_regression_selection_transformer.fit_transform(x_train, y_train.values.ravel()),
        columns=log_regression_selection_transformer.get_feature_names_out(input_features=x_train.columns),
        index=x_train.index,
    )

    print(f"\nSelect From Model by Log Regression:\n{columns[log_regression_selection_transformer.get_support()]}")  # pyright: ignore[reportCallIssue, reportArgumentType]

    return log_regression_selection_df


def _recursive_feature_selection(x_train: pd.DataFrame, columns: pd.Index[str], y_train: pd.DataFrame) -> pd.DataFrame:
    """Use recursive feature selection by log regression"""
    recursive_feature_selection_transformer: RFE = RFE(estimator=LogisticRegression(class_weight="balanced", max_iter=500))
    recursive_feature_selection_df: pd.DataFrame = pd.DataFrame(
        recursive_feature_selection_transformer.fit_transform(x_train, y_train.values.ravel()),
        columns=recursive_feature_selection_transformer.get_feature_names_out(input_features=x_train.columns),
        index=x_train.index,
    )

    print(f"\nRFE by Log Regression:\n{columns[recursive_feature_selection_transformer.get_support()]}")  # pyright: ignore[reportCallIssue, reportArgumentType]

    return recursive_feature_selection_df


def evaluate_text_representations(text_transforms_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    for method, mean_score, std_score in _compare_text_representation(text_transforms_dfs):
        print(f"Text Representation {method} Train: {mean_score:.3f}, Test: {std_score:.3f}")


def _text_principal_component_analysis_selection(text_df: pd.DataFrame, title: str) -> pd.DataFrame:
    """
    Select Features from Text Using Principal Component Analysis (PCA)
    n.b. sBERT embeddings are dense and scaled
    """
    principal_component_analysis_transformer: PCA = PCA()
    principal_component_analysis: pd.DataFrame = principal_component_analysis_transformer.fit_transform(text_df)

    variance_thresholds: tuple[int, ...] = (5, 10, 20, 50, 75)
    explained_variance_ratio = principal_component_analysis_transformer.explained_variance_ratio_

    for threshold in variance_thresholds:
        print(f"Variance by top {threshold} components: {explained_variance_ratio[:threshold].sum():.2f}")

    figure = plt.figure()
    ax = figure.add_subplot(111, projection="3d")
    threshold_count: int = 0

    for i in range(100):
        if explained_variance_ratio[:i].sum() >= 0.8 and not threshold_count:
            print(f"80% of variance explained by {i} components (of {len(text_df.columns)})")
            threshold_count = i

    _ = ax.scatter(
        principal_component_analysis.iloc[:, 0],
        principal_component_analysis.iloc[:, 1],
        principal_component_analysis.iloc[:, 2],  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        alpha=0.1,
        edgecolor="none",
    )

    # How much variance explained along each axis dimension
    _ = ax.set_title(f"{title} sBERT Embedding Dimensional Distribution")
    _ = ax.set_xlabel(f"PCA 1 ({explained_variance_ratio[0]:.1%})")
    _ = ax.set_ylabel(f"PCA 2 ({explained_variance_ratio[1]:.1%})")
    _ = ax.set_zlabel(f"PCA 3 ({explained_variance_ratio[2]:.1%})")

    _ = plt.show()

    return principal_component_analysis


def feature_selection(x_train: pd.DataFrame, y_train: pd.DataFrame, text_transform_df: pd.DataFrame, title: str) -> pd.DataFrame:
    """Select features for final dataset ready for downstream model development"""
    continuous_data_scaler: ColumnTransformer = _get_continuous_data_scaler(x_train)
    scaled_x_train: pd.DataFrame = pd.DataFrame(continuous_data_scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    _ = _variance_threshold_selection(scaled_x_train, x_train.columns)
    _ = _log_regression_selection(scaled_x_train, x_train.columns, y_train)
    _ = _selectkbest_selection(scaled_x_train, x_train.columns, y_train)
    rfe_df: pd.DataFrame = _recursive_feature_selection(scaled_x_train, x_train.columns, y_train)
    text_df: pd.DataFrame = _text_principal_component_analysis_selection(text_transform_df, title)

    print(f"RFE: {rfe_df.shape}")
    print(f"Text PCA: {text_df.shape}")

    # Text DF contains ALL records (need to drop records for physical subset)
    return rfe_df.join(text_df, how="inner")
