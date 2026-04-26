"""Регрессия для предсказания SI."""

from __future__ import annotations

from regression_utils import run_regression_task


def main() -> None:
    results = run_regression_task(target_column="SI", experiment_name="si")
    print("\nСравнение моделей для SI (сортировка по RMSE):")
    print(results[["model", "cv_rmse", "RMSE", "MAE", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
