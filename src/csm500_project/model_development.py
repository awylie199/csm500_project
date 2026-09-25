"""Run and Evaluate Models Across the Feature Development Pipeline"""

from collections.abc import Callable
from typing import Any, Final

import pandas as pd
from imblearn.pipeline import Pipeline
from joblib import Memory
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .util import print_calibration_display, print_confusion_matrix, print_precision_recall_display, print_roc_curve_display

_SEED: Final[int] = 20260920
_MAX_ITERATIONS: Final[int] = 1
_MAX_DEPTH: Final[int] = 20

_MODELS: dict[str, BaseEstimator] = {
    "Log Regression": LogisticRegression(solver="newton-cholesky", tol=1e-1, random_state=_SEED, max_iter=_MAX_ITERATIONS),
    "SGD Classifier": SGDClassifier(loss="log_loss", max_iter=_MAX_ITERATIONS, tol=1e-1, eta0=0.1, random_state=_SEED),
    "K-Neighbours Classifier": KNeighborsClassifier(n_jobs=-1),
    "Decision Tree Classifier": DecisionTreeClassifier(max_depth=_MAX_DEPTH, min_samples_leaf=50, criterion="gini", random_state=_SEED),
    "Linear SVC": LinearSVC(tol=1e-1, max_iter=_MAX_ITERATIONS, random_state=_SEED),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(tol=1e-1),
}

_SCORING_METRICS: Final[dict[str, tuple[str, str]]] = {
    "Accuracy": ("accuracy", "accuracy"),
    "Balanced Accuracy": ("balanced_accuracy", "balanced_accuracy"),
    "F1 Macro": ("f1_macro", "f1_macro"),
}

_PROBABILISTIC_SCORING_METRICS: Final[dict[str, tuple[str, str]]] = {
    **_SCORING_METRICS,
    "ROC AUC": ("roc_auc", "roc_auc_ovr"),
}


def _debug_pipeline_step(pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, step: int) -> None:
    pipeline.fit(x_train, y_train.values.ravel())
    feature_names = x_train.columns.to_numpy()

    for name, transformer in pipeline.steps[:step]:
        feature_names = transformer.get_feature_names_out(feature_names)
        print(f"\n{name}:")
        print(feature_names.tolist())


def fit_model_and_estimate(
    pipeline: Pipeline,
    model: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    model_name: str,
) -> None:
    """Fit and Transform Model on Pipeline, and Estimate on Test"""

    # _debug_pipeline_step(pipeline, x_train, y_train, 8)
    model_scoring_metrics: dict[str, tuple[str, str]] = (
        _PROBABILISTIC_SCORING_METRICS if hasattr(model, "predict_proba") else _SCORING_METRICS
    )
    stratified_kfold: StratifiedKFold = StratifiedKFold(n_splits=3, shuffle=True, random_state=_SEED)

    cross_validate_result = cross_validate(
        pipeline,
        x_train,
        y_train.values.ravel(),
        cv=stratified_kfold,
        n_jobs=-1,
        scoring={k: v for k, v in model_scoring_metrics.values()},
        return_train_score=False,
    )

    for metric_name, metric in model_scoring_metrics.items():
        mean: float = cross_validate_result[f"test_{metric[0]}"].mean()
        std: float = cross_validate_result[f"test_{metric[0]}"].std()
        print(f"{model_name} Test Cross Val {metric_name}: {mean:.4f} ({std:.4f})")

    # Estimate on Test DFs
    pipeline.fit(x_train, y_train.values.ravel())
    y_pred = pipeline.predict(x_test)

    print(f"Classification Report\n{classification_report(y_test.values.ravel(), y_pred)}")
    print(f"Confusion Matrix\n{confusion_matrix(y_test.values.ravel(), y_pred)}")

    print_confusion_matrix(y_test, y_pred, f"{model_name} Confusion Matrix Display")

    y_test_bin = label_binarize(y_test, classes=pipeline.classes_)

    if hasattr(model, "predict_proba"):
        pipeline_probabilities = pipeline.predict_proba(x_test)
        all_classes_macro_average_precision = average_precision_score(y_test_bin, pipeline_probabilities, average="macro")

        print_calibration_display(pipeline.classes_, pipeline_probabilities, y_test, f"{model_name} Calibration")
        print_roc_curve_display(pipeline.classes_, pipeline_probabilities, y_test, f"{model_name} ROC Curve")
        print_precision_recall_display(pipeline.classes_, pipeline_probabilities, y_test, f"{model_name} Precision Recall", True)

    else:
        pipeline_scores = pipeline.decision_function(x_test)
        all_classes_macro_average_precision = average_precision_score(y_test_bin, pipeline_scores, average="macro")

        print_precision_recall_display(pipeline.classes_, pipeline_scores, y_test, f"{model_name} Precision Recall", False)

    print(f"Macro Average Precision Score: {all_classes_macro_average_precision:.3f}")


def run_model_pipeline(
    build_model_pipeline: Callable[[BaseEstimator, Memory], Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> list[Pipeline]:
    """Run the model pipeline across models"""

    run_model_pipeline_memory: Memory = Memory("./model-cache", verbose=0)
    pipelines: list[Pipeline] = []

    for model_name, model in _MODELS.items():
        pipeline: Pipeline = build_model_pipeline(model, run_model_pipeline_memory)

        fit_model_and_estimate(
            pipeline,
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            model_name=model_name,
        )

        pipelines.append(pipeline)

    return pipelines
