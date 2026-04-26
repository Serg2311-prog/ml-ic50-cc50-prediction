"""Классификация для задачи CC50 > median."""

from __future__ import annotations

from classification_utils import ClassificationTask, run_classification_task
from ml_common import load_dataset


def main() -> None:
    df = load_dataset()
    cc50_median = float(df["CC50, mM"].median())
    task = ClassificationTask(
        name="cc50_gt_median",
        source_target_column="CC50, mM",
        threshold=cc50_median,
    )
    result = run_classification_task(task)
    print("\n=== Classification CC50 > median ===")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
