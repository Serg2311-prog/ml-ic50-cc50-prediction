"""Общие entrypoint-функции для запуска экспериментов.

Модуль нужен только для устранения дублирования в файлах запуска.
Логика обучения моделей не меняется: используются те же утилиты,
гиперпараметры и метрики, что и раньше.
"""

from __future__ import annotations

from classification_utils import ClassificationTask, run_classification_task
from ml_common import load_dataset
from regression_utils import run_regression_task


def run_regression_entrypoint(
    *,
    target_column: str,
    experiment_name: str,
    pretty_name: str,
) -> None:
    """Запускает регрессию для одного таргета и печатает таблицу метрик."""
    result_table = run_regression_task(
        target_column=target_column,
        experiment_name=experiment_name,
    )
    print(f"\nСравнение моделей для {pretty_name} (сортировка по RMSE):")
    print(result_table.to_string(index=False))


def run_classification_by_median_entrypoint(
    *,
    source_target_column: str,
    task_name: str,
    pretty_name: str,
) -> None:
    """Запускает классификацию с порогом по медиане выбранного таргета."""
    dataset = load_dataset()
    threshold = float(dataset[source_target_column].median())
    task = ClassificationTask(
        name=task_name,
        source_target_column=source_target_column,
        threshold=threshold,
    )
    result_table = run_classification_task(task)
    print(f"\n=== Классификация: {pretty_name} > median ===")
    print(result_table.to_string(index=False))
