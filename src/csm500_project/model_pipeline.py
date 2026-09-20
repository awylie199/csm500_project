"""Build a model pipeline"""

from collections.abc import Sequence

from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures, PowerTransformer, StandardScaler, TargetEncoder

from .transformer import MulticollinearityDropTransformer, MutualInfoScoresTransformer, SentenceTransformerWrapper


def _get_continuous_data_transformers(continuous_data_columns: Sequence[str]) -> list[tuple[str, ColumnTransformer]]:
    """Get continuous data transformers"""
    box_cox_transformer: ColumnTransformer = ColumnTransformer(
        transformers=[("box_cox", PowerTransformer(method="box-cox", standardize=False), continuous_data_columns)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return [("continuous_data_transforms", box_cox_transformer)]


def _get_drop_columns_transformers(drop_columns: Sequence[str]) -> list[tuple[str, ColumnTransformer]]:
    """Get drop columns transformers (drops specified columns from the dataframe)"""
    return [("drop_columns_transform", ColumnTransformer(transformers=[("drop", "drop", drop_columns)], remainder="passthrough"))]


def _get_discrete_data_transformers(
    label_encoded_columns: Sequence[str], target_encoded_columns: Sequence[str], dummy_encoded_categories: Sequence[str]
) -> list[tuple[str, ColumnTransformer]]:
    """Get discrete (categorical) transformers"""
    return [
        (
            "discrete_data_transformers",
            ColumnTransformer(
                transformers=[
                    ("ordinal_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), label_encoded_columns),
                    (
                        "target_encoder",
                        TargetEncoder(target_type="multiclass", cv=5, smooth="auto"),
                        target_encoded_columns,
                    ),
                    ("onehot_encoder", OneHotEncoder(sparse_output=False, drop="first"), dummy_encoded_categories),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ),
        )
    ]


def _get_polynomial_features_transformers() -> list[tuple[str, ColumnTransformer]]:
    """Get polynomial features transformers"""

    exclude_text_column = make_column_selector(pattern="^(?!Description)")

    return [
        (
            "polynomial_features_transformers",
            ColumnTransformer(
                transformers=[
                    (
                        "polynomial_features",
                        Pipeline(
                            [
                                ("polynomial_features", PolynomialFeatures(include_bias=False)),
                                ("mutual_info_scores_drop", MutualInfoScoresTransformer(0.01)),
                                ("multicollinear_scores_drop", MulticollinearityDropTransformer()),
                            ]
                        ),
                        exclude_text_column,
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ),
        )
    ]


def _get_text_data_transformers() -> list[tuple[str, ColumnTransformer]]:
    """Get text data transformers"""

    return [
        (
            "text_data_transformers",
            ColumnTransformer(
                transformers=[
                    (
                        "text",
                        Pipeline(
                            [
                                ("sbert", SentenceTransformerWrapper()),
                                ("pca", PCA(n_components=0.8)),
                            ]
                        ),
                        ["Description"],
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ),
        )
    ]


def _get_feature_selection_transformers() -> list[tuple[str, ColumnTransformer]]:
    """Get feature selection transformers"""

    include_all_numeric_columns = make_column_selector(pattern="Floor Area (m^2)")
    exclude_text_column = make_column_selector(pattern="^(?!Description)")

    return [
        (
            "feature_selection_transformers",
            ColumnTransformer(
                transformers=[
                    ("numeric_standard_scaler", StandardScaler(), include_all_numeric_columns),
                    ("variance_threshold", VarianceThreshold(), exclude_text_column),
                    (
                        "recursive_feature_selection",
                        RFE(estimator=LogisticRegression(class_weight="balanced", max_iter=500)),
                        exclude_text_column,
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ),
        )
    ]


def build_model_pipeline(
    continuous_data_columns: Sequence[str],
    drop_columns: Sequence[str],
    label_encoded_categories: Sequence[str],
    target_encoded_categories: Sequence[str],
    dummy_encoded_categories: Sequence[str],
) -> Pipeline:
    """
    Build a model pipeline to mitigate leakage and automate transformation
    # Map
    # Drop Uncorrelated Columns
    # Select Final Features
    """

    return Pipeline(
        [
            *_get_continuous_data_transformers(continuous_data_columns),
            *_get_drop_columns_transformers(drop_columns),
            *_get_discrete_data_transformers(label_encoded_categories, target_encoded_categories, dummy_encoded_categories),
            *_get_polynomial_features_transformers(),
            *_get_text_data_transformers(),
            *_get_feature_selection_transformers(),
        ]
    )
