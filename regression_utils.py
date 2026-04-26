"""Пайплайн регрессионных экспериментов для курсового проекта."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_common import (
    RANDOM_STATE,
    build_regression_frame,
    make_results_subdir,
    regression_metrics,
    split_data,
)


CV_FOLDS = 5
TREE_SEARCH_ITER = 16


@dataclass(frozen=True)
class SearchConfig:
    model_name: str
    pipeline: Pipeline
    params: dict[str, list]
    search_kind: str  # grid | random


def _build_search_configs() -> list[SearchConfig]:
    linear_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    rf_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    gbr_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]
    )

    return [
        SearchConfig(
            model_name="Linear Regression",
            pipeline=linear_pipeline,
            params={
                "model__fit_intercept": [True, False],
                "model__positive": [False, True],
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
            },
            search_kind="random",
        ),
        SearchConfig(
            model_name="Gradient Boosting",
            pipeline=gbr_pipeline,
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


def run_regression_task(target_column: str, experiment_name: str) -> pd.DataFrame:
    """Запуск полного цикла регрессии для указанного таргета."""
    x, y = build_regression_frame(target_column)
    x_train, x_test, y_train, y_test = split_data(x, y, stratify=False)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    exp_dir = make_results_subdir(f"regression_{experiment_name}")

    records: list[dict[str, float | str]] = []
    best_rmse = float("inf")
    best_predictions: np.ndarray | None = None
    best_true: np.ndarray | None = None
    for config in _build_search_configs():
        if config.search_kind == "grid":
            search = GridSearchCV(
                estimator=config.pipeline,
                param_grid=config.params,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
        else:
            search = RandomizedSearchCV(
                estimator=config.pipeline,
                param_distributions=config.params,
                n_iter=TREE_SEARCH_ITER,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                refit=True,
            )

        search.fit(x_train, y_train)
        best_model = search.best_estimator_
        predictions = best_model.predict(x_test)
        metrics = regression_metrics(y_test, predictions)
        records.append(
            {
                "target": target_column,
                "model": config.model_name,
                "cv_rmse": float(abs(search.best_score_)),
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "R2": metrics["R2"],
                "best_params": json.dumps(search.best_params_, ensure_ascii=False),
            }
        )
        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_predictions = np.asarray(predictions)
            best_true = y_test.to_numpy()

    result_df = pd.DataFrame(records).sort_values(by="RMSE", ascending=True)
    result_df.to_csv(exp_dir / "model_comparison.csv", index=False)

    best_row = result_df.iloc[0]
    summary_lines = [
        f"# Регрессия: {target_column}",
        "",
        f"- Размер train/test: {x_train.shape[0]}/{x_test.shape[0]}",
        f"- Число признаков: {x_train.shape[1]}",
        f"- Лучшая модель: **{best_row['model']}**",
        (
            f"- Метрики лучшей модели: RMSE={best_row['RMSE']:.4f}, "
            f"MAE={best_row['MAE']:.4f}, R2={best_row['R2']:.4f}"
        ),
        "",
        "## Интерпретация",
        (
            "Лучшей считалась модель с минимальным RMSE на тестовой выборке, "
            "поскольку в задачах токсичности и активности важно минимизировать "
            "крупные ошибки прогноза."
        ),
        (
            "Дополнительно стоит проверить устойчивость на внешнем датасете и "
            "рассмотреть feature engineering (группы дескрипторов, PCA, отбор по SHAP)."
        ),
    ]
    (exp_dir / "conclusions.md").write_text("\n".join(summary_lines), encoding="utf-8")

    if best_predictions is None or best_true is None:
        raise RuntimeError("Не удалось определить лучшую модель регрессии.")

    preview = pd.DataFrame({"y_true": best_true, "y_pred": best_predictions})
    preview.to_csv(exp_dir / "best_model_predictions.csv", index=False)
    return result_df
