"""Регрессионное моделирование для прогноза IC50."""

from __future__ import annotations

from regression_utils import run_regression_task


def main() -> None:
    results = run_regression_task(target_column="IC50, mM", experiment_name="ic50")
    print("\nСравнение моделей для IC50 (сортировка по RMSE):")
    print(results.to_string(index=False))
    print("\nФайлы с результатами сохранены в: results/regression_ic50")


if __name__ == "__main__":
    main()
