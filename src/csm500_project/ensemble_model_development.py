"""Final Ensemble Model Fitting and Selection"""

from collections.abc import Callable, Sequence
from typing import Final

import pandas as pd
from imblearn.pipeline import Pipeline
from joblib import Memory
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize

from .util import print_calibration_display, print_confusion_matrix, print_precision_recall_display, print_roc_curve_display

_SEED: Final[int] = 20260924


def run_kneighbours_bagging_classifier_pipeline(
    build_model_pipeline: Callable[[BaseEstimator, Memory | None], Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Pipeline:
    kneighbours_classifier: KNeighborsClassifier = KNeighborsClassifier(n_jobs=-1)
    bagging_classifier: BaggingClassifier = BaggingClassifier(
        estimator=kneighbours_classifier, bootstrap=True, n_jobs=-1, random_state=_SEED
    )
    pipeline: Pipeline = build_model_pipeline(bagging_classifier, None)
    stratified_kfold: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=_SEED)
    print("HER")
    bagging_classifier_k_neighbours: RandomizedSearchCV = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={
            "model__estimator__n_neighbors": [10, 20, 30],
            "model__estimator__weights": ["uniform", "distance"],
            "model__estimator__metric": ["minkowski", "euclidean", "manhattan"],
            "model__n_estimators": [10, 25, 50, 100],
            "model__max_samples": [0.5, 0.75, 1.0],
            "model__max_features": [0.5, 0.75, 1.0],
        },
        cv=stratified_kfold,
        scoring="f1_macro",
        return_train_score=False,
        n_jobs=-1,
        refit=True,
        random_state=_SEED,
        n_iter=20,
    )
    bagging_classifier_k_neighbours.fit(x_train, y_train.values.ravel())
    y_pred = bagging_classifier_k_neighbours.predict(x_test)
    pipeline_probabilities = bagging_classifier_k_neighbours.predict_proba(x_test)
    y_test_bin = label_binarize(y_test, classes=bagging_classifier_k_neighbours.classes_)
    all_classes_macro_average_precision = average_precision_score(y_test_bin, pipeline_probabilities, average="macro")

    print(f"Bagging Classifier Best Estimator: {bagging_classifier_k_neighbours.best_estimator_}")
    print(f"Bagging Classifier Best Params: {bagging_classifier_k_neighbours.best_params_}")
    print(f"Bagging Classifier Best Score: {bagging_classifier_k_neighbours.best_score_}")
    print(f"Classification Report\n{classification_report(y_test.values.ravel(), y_pred)}")
    print(f"Confusion Matrix\n{confusion_matrix(y_test.values.ravel(), y_pred)}")
    print(f"Macro Average Precision Score: {all_classes_macro_average_precision:.3f}")

    print_confusion_matrix(y_test, y_pred, f"K-Neighbours Bagging Classifier Confusion Matrix Display")
    print_calibration_display(
        bagging_classifier_k_neighbours.classes_, pipeline_probabilities, y_test, "K-Neighbours Bagging Classifier Calibration"
    )
    print_roc_curve_display(
        bagging_classifier_k_neighbours.classes_, pipeline_probabilities, y_test, "K-Neighbours Bagging Classifier ROC Curve"
    )
    print_precision_recall_display(
        bagging_classifier_k_neighbours.classes_, pipeline_probabilities, y_test, "K-Neighbours Bagging Classifier Precision Recall", True
    )

    return pipeline
