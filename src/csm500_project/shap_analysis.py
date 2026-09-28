"""Perform SHAP Analysis on Best Performing Pipeline"""

from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.pipeline import Pipeline

_RANDOM_STATE: Final[int] = 20260926


def _map_class_to_name(target_class: int) -> str:
    """Map Class Index to Descriptive Name (Inferred from Value Distribution)"""
    if target_class == 0:
        return "Approve"
    elif target_class == 1:
        return "Refuse"
    elif target_class == 2:
        return "Split"
    else:
        raise ValueError(f"unknown target class: {target_class}")


def run_shap_analysis(pipeline: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame) -> None:
    """Run SHAP Analysis on Pipeline"""

    preprocessing: Pipeline = pipeline[:-2]  # Exclude model and SMOTE
    model = pipeline.named_steps["model"]

    x_train_transformed: pd.DataFrame = preprocessing.transform(x_train)
    x_test_transformed: pd.DataFrame = preprocessing.transform(x_test)
    feature_names = preprocessing.get_feature_names_out()

    x_train_transformed: pd.DataFrame = pd.DataFrame(x_train_transformed, columns=feature_names, index=x_train.index)
    x_test_transformed: pd.DataFrame = pd.DataFrame(x_test_transformed, columns=feature_names, index=x_test.index)

    sample_x_train: pd.DataFrame = shap.sample(x_train_transformed, 50, random_state=_RANDOM_STATE)
    sample_x_test: pd.DataFrame = shap.sample(x_test_transformed, 50, random_state=_RANDOM_STATE)
    explainer: shap.Explainer = shap.Explainer(model.predict_proba, sample_x_train)
    shap_results: shap.Explanation | list[shap.Explanation] = explainer(sample_x_test)

    for i, class_name in enumerate(model.classes_):
        print(f"SHAP explanation for class: {_map_class_to_name(class_name)}")
        class_shap_results: shap.Explanation = shap_results[..., i]  # pyright: ignore[reportCallIssue, reportArgumentType]

        shap_df: pd.DataFrame = pd.DataFrame(class_shap_results.values, columns=x_test_transformed.columns)
        sbert_columns_indices: list[str] = [c for c in shap_df.columns if c.startswith("pca")]
        shap_df["Property Description (sBERT)"] = shap_df[sbert_columns_indices].sum(axis=1)
        shap_df: pd.DataFrame = shap_df.drop(columns=sbert_columns_indices)

        aggregated_shap: shap.Explanation = shap.Explanation(
            values=shap_df.values, base_values=class_shap_results.base_values, feature_names=shap_df.columns.tolist()
        )

        shap.plots.beeswarm(aggregated_shap, max_display=15)
        shap.plots.bar(aggregated_shap, max_display=15)
        shap.plots.waterfall(aggregated_shap[0])  # sBERT Embeddings


def _plot_missclassified_rate(anomalies_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 8))
    plt.bar(anomalies_df.index.astype(str), anomalies_df["error_rate_pct"])

    plt.xlabel("Prediction Confidence")
    plt.ylabel("Misclassification Rate (%)")
    plt.title("Misclassification Rate by Prediction Confidence")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def _plot_missclassified_observations(anomalies_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 8))

    plt.bar(anomalies_df.index.astype(str), anomalies_df["total"])

    plt.xlabel("Prediction Confidence")
    plt.ylabel("Number of Observations")
    plt.title("Number of Predictions by Confidence")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def run_shap_inconsistency_analysis(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """Run SHAP to Evaluate Confident Anomalies"""
    y_pred = pipeline.predict(x_test)
    y_pred_2d = np.asarray(y_pred).ravel()
    y_test_2d = np.asarray(y_test).ravel()
    y_probabilities = pipeline.predict_proba(x_test)
    confidence = y_probabilities.max(axis=1)
    anomalies_df: pd.DataFrame = pd.DataFrame({"misclassified": y_pred_2d != y_test_2d, "confidence": confidence}, index=x_test.index)

    bins: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1)
    anomalies_df["confidence_band"] = pd.cut(anomalies_df["confidence"], bins=bins, include_lowest=True)
    anomalies_summary_df: pd.DataFrame = anomalies_df.groupby("confidence_band", observed=False).agg(
        total=("misclassified", "size"), errors=("misclassified", "sum"), error_rate=("misclassified", "mean")
    )
    anomalies_summary_df["error_rate_pct"] = anomalies_summary_df["error_rate"] * 100

    print(anomalies_summary_df)

    _plot_missclassified_rate(anomalies_summary_df)
    _plot_missclassified_observations(anomalies_summary_df)
