"""Пайплайн классификационных экспериментов для курсового проекта."""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_common import (
    RANDOM_STATE,
    assert_no_target_like_columns,
    build_binary_target,
    build_feature_matrix,
    load_dataset,
    make_results_subdir,
    split_data,
)


CV_FOLDS = 5
TREE_SEARCH_ITER = 16


@dataclass(frozen=True)
class ClassificationTask:
    name: str
    source_target_column: str
    threshold: float


@dataclass(frozen=True)
class SearchConfig:
    model_name: str
    pipeline: Pipeline
    params: dict[str, list]
    search_kind: str  # grid | random


def _build_search_configs() -> list[SearchConfig]:
    logreg_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=3000,
                    solver="liblinear",
                ),
            ),
        ]
    )
    rf_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    gbc_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )

    return [
        SearchConfig(
            model_name="Logistic Regression",
            pipeline=logreg_pipeline,
            params={
                "model__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
                "model__penalty": ["l1", "l2"],
                "model__class_weight": [None, "balanced"],
            },
            search_kind="grid",
        ),
        SearchConfig(
            model_name="Random Forest",
            pipeline=rf_pipeline,
            params={
                "model__n_estimators": [250, 400, 600],
                "model__max_depth": [None, 8, 16, 24],
                "model__min_samples_split": [2, 4, 8],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.5, 0.8],
                "model__class_weight": [None, "balanced"],
            },
            search_kind="random",
        ),
        SearchConfig(
            model_name="Gradient Boosting",
            pipeline=gbc_pipeline,
            params={
                "model__n_estimators": [100, 200, 300, 500],
                "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 4, 5],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__subsample": [0.7, 0.85, 1.0],
            },
            search_kind="random",
        ),
    ]


def _prepare_task_data(task: ClassificationTask) -> tuple[pd.DataFrame, pd.Series]:
    df = load_dataset().copy()
    df = df.dropna(subset=[task.source_target_column])
    x = build_feature_matrix(df, target_column=task.source_target_column)
    y = build_binary_target(df[task.source_target_column], threshold=task.threshold)
    return x, y


def _save_confusion_matrix_figure(matrix: np.ndarray, labels: list[str], path: str) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def run_classification_task(task: ClassificationTask) -> pd.DataFrame:
    """Запуск полного цикла классификации для одной постановки."""
    x, y = _prepare_task_data(task)
    x_train, x_test, y_train, y_test = split_data(x, y, stratify=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    exp_dir = make_results_subdir(f"classification_{task.name}")

    records: list[dict[str, float | str]] = []
    best_model_name = ""
    best_auc = -1.0
    best_predictions: np.ndarray | None = None
    best_probabilities: np.ndarray | None = None

    for config in _build_search_configs():
        assert_no_target_like_columns(x_train)
        if config.search_kind == "grid":
            search = GridSearchCV(
                estimator=config.pipeline,
                param_grid=config.params,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
        else:
            search = RandomizedSearchCV(
                estimator=config.pipeline,
                param_distributions=config.params,
                n_iter=TREE_SEARCH_ITER,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                refit=True,
            )

        search.fit(x_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(x_test)
        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(x_test)[:, 1]
        else:
            y_proba = best_model.decision_function(x_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_proba))
        records.append(
            {
                "task": task.name,
                "model": config.model_name,
                "cv_roc_auc": float(search.best_score_),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": auc,
                "best_params": json.dumps(search.best_params_, ensure_ascii=False),
            }
        )

        if auc > best_auc:
            best_auc = auc
            best_model_name = config.model_name
            best_predictions = y_pred
            best_probabilities = y_proba

    result_df = pd.DataFrame(records).sort_values(by="roc_auc", ascending=False)
    result_df.to_csv(exp_dir / "model_comparison.csv", index=False)

    if best_predictions is None or best_probabilities is None:
        raise RuntimeError("Не удалось определить лучшую модель классификации.")

    conf = confusion_matrix(y_test, best_predictions)
    _save_confusion_matrix_figure(conf, labels=["0", "1"], path=str(exp_dir / "best_confusion_matrix.png"))

    prediction_table = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": best_predictions,
            "y_proba": best_probabilities,
        }
    )
    prediction_table.to_csv(exp_dir / "best_model_predictions.csv", index=False)

    best_row = result_df.iloc[0]
    summary_lines = [
        f"# Классификация: {task.name}",
        "",
        f"- Порог: **{task.threshold:.4f}** для колонки `{task.source_target_column}`.",
        f"- Размер train/test: {x_train.shape[0]}/{x_test.shape[0]}",
        f"- Число признаков: {x_train.shape[1]}",
        f"- Лучшая модель по ROC-AUC: **{best_model_name}**",
        (
            f"- Метрики лучшей модели: accuracy={best_row['accuracy']:.4f}, "
            f"precision={best_row['precision']:.4f}, recall={best_row['recall']:.4f}, "
            f"F1={best_row['f1']:.4f}, ROC-AUC={best_row['roc_auc']:.4f}"
        ),
        "",
        "## Интерпретация",
        (
            "ROC-AUC использована как основная метрика, потому что в задачах скрининга "
            "соединений важно отделять активные молекулы при разных порогах принятия решения."
        ),
        (
            "Для следующего шага полезно откалибровать вероятности (Platt/Isotonic), "
            "чтобы уверенность модели точнее отражала реальную вероятность класса."
        ),
    ]
    (exp_dir / "conclusions.md").write_text("\n".join(summary_lines), encoding="utf-8")
    return result_df
