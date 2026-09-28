"""Final Ensemble Model Fitting and Selection"""

from collections.abc import Callable, Sequence
from typing import Final

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline
from joblib import Memory
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

from .constant import ADABOOST_DECISION_TREE_DUMP, BAGGING_CLASSIFIER_KNEIGHBOURS_DUMP, RANDOM_FOREST_FILE_DUMP
from .model_development import PROBABILISTIC_SCORING_METRICS as SCORING_METRICS
from .util import print_calibration_display, print_confusion_matrix, print_precision_recall_display, print_roc_curve_display

_SEED: Final[int] = 20260924
_N_ITERATIONS: Final[int] = 10

ensemble_model_scoring = {k: v for k, v in SCORING_METRICS.values()}


def _get_stratified_k_fold() -> StratifiedKFold:
    """Get Experiment Stratified K-Fold"""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=_SEED)


def _print_results(name: str, optimised_pipeline: RandomizedSearchCV, x_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """Print Results of Optimised Pipelines"""
    y_pred = optimised_pipeline.predict(x_test)
    pipeline_probabilities = optimised_pipeline.predict_proba(x_test)
    y_test_bin = label_binarize(y_test, classes=optimised_pipeline.classes_)
    all_classes_macro_average_precision = average_precision_score(y_test_bin, pipeline_probabilities, average="macro")

    optimised_results_df: pd.DataFrame = pd.DataFrame(optimised_pipeline.cv_results_)
    best_model_results = optimised_results_df.loc[optimised_pipeline.best_index_]  # pyright: ignore[reportCallIssue, reportArgumentType]

    print(f"{name} Best Estimator: {optimised_pipeline.best_estimator_}")
    print(f"{name}: {optimised_pipeline.best_params_}")
    print(f"{name} Best Score: {optimised_pipeline.best_score_}")
    print(f"{name} Classification Report\n{classification_report(y_test.values.ravel(), y_pred)}")
    print(f"{name} Confusion Matrix\n{confusion_matrix(y_test.values.ravel(), y_pred)}")
    print(f"{name} Macro Average Precision Score: {all_classes_macro_average_precision:.3f}")
    print(f"{name} F1 macro: {best_model_results['mean_test_f1_macro']:.4f} ({best_model_results['std_test_f1_macro']:.4f})")
    print(f"{name} Accuracy: {best_model_results['mean_test_accuracy']:.4f} ({best_model_results['std_test_accuracy']:.4f})")
    print(
        f"{name} Balanced accuracy: {best_model_results['mean_test_balanced_accuracy']:.4f} ({best_model_results['std_test_balanced_accuracy']:.4f})"
    )
    print(f"{name} ROC AUC: {best_model_results['mean_test_roc_auc']:.4f} ({best_model_results['std_test_roc_auc']:.4f})")

    print_confusion_matrix(y_test, y_pred, f"{name} Confusion Matrix Display")
    print_calibration_display(optimised_pipeline.classes_, pipeline_probabilities, y_test, f"{name} Calibration")
    print_roc_curve_display(optimised_pipeline.classes_, pipeline_probabilities, y_test, f"{name} ROC Curve")
    print_precision_recall_display(optimised_pipeline.classes_, pipeline_probabilities, y_test, f"{name} Precision Recall", True)


def run_kneighbours_bagging_classifier_pipeline(
    build_model_pipeline: Callable[[BaseEstimator, Memory | None], Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Pipeline:
    """Run Bagging Classifier with K-Neighbours Optimised Classifier"""
    kneighbours_classifier: KNeighborsClassifier = KNeighborsClassifier(n_jobs=-1)
    bagging_classifier: BaggingClassifier = BaggingClassifier(
        estimator=kneighbours_classifier, bootstrap=True, n_jobs=-1, random_state=_SEED
    )
    pipeline: Pipeline = build_model_pipeline(bagging_classifier, None)
    stratified_kfold: StratifiedKFold = _get_stratified_k_fold()
    optimised_bagging_classifier_k_neighbours: RandomizedSearchCV = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={
            "model__estimator__n_neighbors": randint(10, 30),
            "model__estimator__weights": ["uniform", "distance"],
            "model__estimator__metric": ["minkowski", "euclidean"],
            "model__n_estimators": randint(50, 100),
            "model__max_samples": uniform(0.75, 1.0),
            "model__max_features": uniform(0.5, 0.75),
        },
        cv=stratified_kfold,
        scoring=ensemble_model_scoring,
        refit="f1_macro",  # pyright: ignore[reportArgumentType]
        return_train_score=False,
        n_jobs=-1,
        random_state=_SEED,
        n_iter=_N_ITERATIONS,
    )
    optimised_bagging_classifier_k_neighbours.fit(x_train, y_train.values.ravel())
    _print_results("K-Neighbours Bagging Classifier", optimised_bagging_classifier_k_neighbours, x_test, y_test)
    joblib.dump(optimised_bagging_classifier_k_neighbours, BAGGING_CLASSIFIER_KNEIGHBOURS_DUMP)

    return pipeline


def run_random_forest_pipeline(
    build_model_pipeline: Callable[[BaseEstimator, Memory | None], Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Pipeline:
    """Run Random Forest Optimised Classifier"""
    random_forest_classifier: RandomForestClassifier = RandomForestClassifier(n_jobs=-1, random_state=_SEED)
    stratified_k_fold: StratifiedKFold = _get_stratified_k_fold()
    pipeline: Pipeline = build_model_pipeline(random_forest_classifier, None)
    optimised_random_forest: RandomizedSearchCV = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={
            "model__n_estimators": randint(200, 1000),
            "model__max_depth": randint(25, 50),
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 10),
            "model__max_features": ["sqrt", "log2"],
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__class_weight": ["balanced", "balanced_subsample"],
        },
        cv=stratified_k_fold,
        scoring=ensemble_model_scoring,
        return_train_score=False,
        n_jobs=-1,
        refit="f1_macro",  # pyright: ignore[reportArgumentType]
        random_state=_SEED,
        n_iter=_N_ITERATIONS,
    )

    optimised_random_forest.fit(x_train, y_train.values.ravel())

    _print_results("Random Forest Classifier", optimised_random_forest, x_test, y_test)
    joblib.dump(optimised_random_forest, RANDOM_FOREST_FILE_DUMP)

    return pipeline


def run_adaboost_decision_tree_pipeline(
    build_model_pipeline: Callable[[BaseEstimator, Memory | None], Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Pipeline:
    """Run ADA Boost Decision Tree Optimised Classifier"""

    decision_tree_classifier: DecisionTreeClassifier = DecisionTreeClassifier(random_state=_SEED)
    adaboost_classifier: AdaBoostClassifier = AdaBoostClassifier(decision_tree_classifier, random_state=_SEED)
    stratified_k_fold: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=_SEED)
    pipeline: Pipeline = build_model_pipeline(adaboost_classifier, None)
    optimised_adaboost_decision_tree: RandomizedSearchCV = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={
            "model__estimator__max_depth": randint(1, 5),
            "model__estimator__min_samples_leaf": randint(1, 10),
            "model__n_estimators": randint(50, 500),
            "model__learning_rate": uniform(0.01, 1.0),
        },
        cv=stratified_k_fold,
        scoring=ensemble_model_scoring,
        return_train_score=False,
        n_jobs=-1,
        refit="f1_macro",  # pyright: ignore[reportArgumentType]
        random_state=_SEED,
        n_iter=_N_ITERATIONS,
    )

    optimised_adaboost_decision_tree.fit(x_train, y_train.values.ravel())

    _print_results("ADA Boost Decision Tree Classifier", optimised_adaboost_decision_tree, x_test, y_test)
    joblib.dump(optimised_adaboost_decision_tree, ADABOOST_DECISION_TREE_DUMP)

    return pipeline
