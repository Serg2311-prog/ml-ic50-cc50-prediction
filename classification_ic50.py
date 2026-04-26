"""Классификация для задачи IC50 > median."""

from __future__ import annotations

from classification_utils import ClassificationTask, run_classification_task
from ml_common import load_dataset


def main() -> None:
    df = load_dataset()
    threshold = float(df["IC50, mM"].median())
    task = ClassificationTask(
        name="ic50_gt_median",
        source_target_column="IC50, mM",
        threshold=threshold,
    )
    result_df = run_classification_task(task)
    print("=== IC50 classification completed ===")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
