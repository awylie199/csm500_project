"""Analyse and Evaluate Correlation between Independent Explanatory Fields"""

from typing import Any, Final, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Categorical
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures

from .constant import INFERRED_TARGET_FIELD
from .util import split_primary_physical_datasets

_RANDOM_STATE: Final[int] = 20260908
_PHYSICAL_FIELDS: Final[tuple[str, ...]] = (
    "Floor Area (m^2)",
    "Is Domestic",
    "Property Type",
)

_CORRELATIONS_NAMES: Final[dict[str, str]] = {
    "Development Classification": "Dev Cl.",
    "Decision Date Year Of Transfer": "Y. Tra.",
    "Decision Date Quarter Of Transfer": "Q. Tra.",
    "LSOA Code": "LSOA",
    "IMD Rank": "IMD R.",
    "IMD Decile": "IMD D.",
    "In Conservation Area": "Cons.",
    "Geohazards Collapsible Deposits Susceptibility": "Geo Col.",
    "Geohazards Running Sand Susceptibility": "Geo San.",
    "Geohazards Shrink Swell Susceptibility": "Geo Shr.",
    "Geohazards Soluble Rocks Susceptibility": "Geo Sol.",
    "In Flood Zone": " Flood",
    "Property Type": "Prop. Ty.",
    "Floor Area (m^2)": "Flo. Area",
    "Is Domestic": "Dom.",
    "Near Brownfield Site": "Brow.",
    "In Built Up Area": "BUA",
    "In Article 4 Area": "Art.4",
    "Organisation Entity_26": "O.26",
    "Organisation Entity_382": "O.382",
    "Organisation Entity_90": "O.90",
    "Council Code_E09000003": "C.3",
    "Council Code_E09000007": "C.7",
    "Council Code_E09000014": "C.14",
    "Council Code_E09000019": "C.19",
    "Council Code_E09000033": "C.33",
    "Council Code_E10000032": "C.32",
    "Listing Grade_1": "LG.1",
    "Listing Grade_2": "LG.2",
    "Listing Grade_3": "LG.3",
    "Parish Code_E04009848": "P.848",
    "Parish Code_E43000147": "P.147",
    "Parish Code_E43000150": "P.150",
    "Parish Code_E43000191": "P.191",
    "Parish Code_E43000193": "P.193",
    "Parish Code_E43000197": "P.197",
    "Parish Code_E43000204": "P.204",
    "Parish Code_E43000209": "P.209",
    "Parish Code_E43000236": "P.236",
}


def _print_correlations_matrix(
    correlations_df: pd.DataFrame, names: tuple[str, ...], method: Literal["Spearman", "Pearson"], title: str
) -> None:
    """Print the correlation matrix"""
    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations_df, vmin=-1, vmax=1)
    _ = fig.colorbar(cax)
    _ = plt.title(f"{title} {method} Rank Correlation Matrix")

    ticks = np.arange(0, len(correlations_df.columns), 1)
    _ = ax.set_xticks(ticks)
    _ = ax.set_yticks(ticks)
    _ = ax.set_xticklabels(names, rotation=45)
    _ = ax.set_yticklabels(names, rotation=45)

    plt.tight_layout()

    for (x, y), value in np.ndenumerate(correlations_df):
        _ = ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8)

    plt.show()


def _plot_correlation_matrices(x_train: pd.DataFrame, columns: tuple[str, ...], title: str) -> None:
    """Plot MultiInfo Correlation Matrices"""
    # Use Spearman's and Pearson's to demonstrate inter-collinearity between explanatory variables
    # However, target is multi-class, and is therefore not used.
    correlations_df = x_train.copy(deep=True).reindex(columns=list(columns))
    pearsons_correlation_df: pd.DataFrame = correlations_df.corr(method="pearson")
    # Spearman's Rank Correlation for improved handling of ordinal categories
    spearmans_correlation_df: pd.DataFrame = correlations_df.corr(method="spearman")
    names: tuple[str, ...] = tuple(v for k, v in _CORRELATIONS_NAMES.items() if k in columns)

    _print_correlations_matrix(spearmans_correlation_df, names, "Pearson", title)
    _print_correlations_matrix(pearsons_correlation_df, names, "Spearman", title)


def _plot_mutual_info_heatmap(x_train: pd.DataFrame, y_train: pd.Series, title: str) -> None:
    """Plot a heatmap ranking of mutual info scores for variables and pairwise features"""
    pf_correlation_analysis_df: pd.DataFrame = x_train.copy(deep=True).drop(columns=["Description"])
    polynomial_features_transformer: PolynomialFeatures = PolynomialFeatures(include_bias=False)
    pf_correlation_analysis_df = pd.DataFrame(
        polynomial_features_transformer.fit_transform(pf_correlation_analysis_df),
        columns=polynomial_features_transformer.get_feature_names_out(),
        index=x_train.index,
    )

    mi_scores = mutual_info_classif(pf_correlation_analysis_df, y_train, random_state=42)
    mi_series: pd.Series = pd.Series(mi_scores, index=pf_correlation_analysis_df.columns).sort_values(ascending=False).head(15)
    mi_matrix = mi_series.values.reshape(-1, 1)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

    _, ax = plt.subplots(figsize=(6, 8))
    _ = ax.imshow(mi_matrix, cmap="Blues", aspect="auto")
    _ = ax.set_yticks(range(len(mi_series)))
    _ = ax.set_yticklabels(mi_series.index, fontsize=10)
    _ = ax.set_xticks([])

    # Annotate numerical values inside the cells
    for i in range(len(mi_series)):
        score: Categorical | Any = mi_matrix[i, 0]
        text_color = "white" if score > mi_matrix.max() / 2 else "black"  # pyright: ignore[reportUnknownMemberType, reportOperatorIssue]
        _ = ax.text(0, i, f"{score:.3f}", ha="center", va="center", color=text_color, fontsize=10)

    _ = plt.title(f"{title} Top 15 Mutual Info Scores", fontsize=12, pad=15)
    plt.tight_layout()
    plt.show()


def _print_text_correlations(text_transforms_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    """Print Correlations Between Text Encodings and Target"""

    count_vectorizer_df, tfidf_vectorizer_df, _ = text_transforms_dfs

    count_vectorizer_correlations_df: pd.DataFrame = tfidf_vectorizer_df.corr()
    tfidf_correlations_df: pd.DataFrame = count_vectorizer_df.corr()

    print("Count Vectorizer Correlations:")
    print(count_vectorizer_correlations_df[INFERRED_TARGET_FIELD].sort_values(ascending=False).to_string())
    print("-------------")
    print("TFID Correlations:")
    print(tfidf_correlations_df[INFERRED_TARGET_FIELD].sort_values(ascending=False).to_string())


def correlation_analysis(
    x_train: pd.DataFrame, y_train: pd.DataFrame, text_transforms_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
) -> None:
    """Correlation Analysis for Pre-Feature Selection"""
    print(x_train.columns)
    print(y_train.columns)

    datasets = split_primary_physical_datasets(x_train, y_train)
    primary_columns: tuple[str, ...] = tuple([c for c in _CORRELATIONS_NAMES if c not in _PHYSICAL_FIELDS])
    _plot_correlation_matrices(datasets["x_primary"], primary_columns, "Primary Dataset")
    _plot_correlation_matrices(datasets["x_physical"], tuple(_CORRELATIONS_NAMES.keys()), "Physical Dataset")
    _plot_mutual_info_heatmap(datasets["x_primary"], datasets["y_primary"][INFERRED_TARGET_FIELD], "Primary Dataset")
    _plot_mutual_info_heatmap(datasets["x_physical"], datasets["y_physical"][INFERRED_TARGET_FIELD], "Physical Dataset")

    _print_text_correlations(text_transforms_dfs)
