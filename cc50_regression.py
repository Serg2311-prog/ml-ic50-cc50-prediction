"""Эксперимент по регрессии CC50."""

from __future__ import annotations

from regression_utils import run_regression_task


def main() -> None:
    results = run_regression_task(target_column="CC50, mM", experiment_name="cc50")
    print("\n=== CC50 regression: model comparison ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
