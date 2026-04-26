"""Классификация для задач SI > median и SI > 8."""

from __future__ import annotations

from classification_utils import ClassificationTask, run_classification_task
from ml_common import load_dataset


def run_si_tasks() -> None:
    df = load_dataset()
    si_median = float(df["SI"].median())
    tasks = [
        ClassificationTask(
            name="si_gt_median",
            source_target_column="SI",
            threshold=si_median,
        ),
        ClassificationTask(
            name="si_gt_8",
            source_target_column="SI",
            threshold=8.0,
        ),
    ]
    for task in tasks:
        result_df = run_classification_task(task)
        print(f"\n[RESULT] {task.name}")
        print(result_df.to_string(index=False))


if __name__ == "__main__":
    run_si_tasks()
