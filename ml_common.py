"""Общие утилиты для курсового проекта по классическому ML."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMNS = ("IC50, mM", "CC50, mM", "SI")
TARGET_LIKE_COLUMNS = (
    "IC50, mM",
    "CC50, mM",
    "SI",
    "IC50_log",
    "CC50_log",
    "SI_log",
)
SI_TARGET_LIKE_NAMES = {
    "si",
    "si_log",
    "log_si",
    "si_bin",
    "si_class",
    "si_scaled",
    "target_si",
}
DATA_FILE = Path("data") / "Данные_для_курсовой_Классическое_МО.xlsx"


def project_root() -> Path:
    return Path(__file__).resolve().parent


def data_path() -> Path:
    return project_root() / DATA_FILE


def results_dir() -> Path:
    path = project_root() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset() -> pd.DataFrame:
    """Чтение исходного датасета и удаление служебного индекса."""
    df = pd.read_excel(data_path())
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def get_target_like_columns(columns: Iterable[str]) -> list[str]:
    """Возвращает endpoint/target-like колонки и известные производные таргетов."""
    target_like: list[str] = []
    explicit_targets = set(TARGET_LIKE_COLUMNS)
    for col in columns:
        normalized = col.lower()
        if (
            col in explicit_targets
            or "ic50" in normalized
            or "cc50" in normalized
            or normalized in SI_TARGET_LIKE_NAMES
        ):
            target_like.append(col)
    return target_like


def build_feature_matrix(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Формирует descriptor-only матрицу признаков без утечки таргетов.

    Все endpoint/target-like колонки и их производные удаляются для любой задачи.
    """
    _ = target_column
    cols_to_drop = get_target_like_columns(df.columns)
    return df.drop(columns=cols_to_drop)


def validate_no_target_like_columns(x: pd.DataFrame) -> None:
    """Выбрасывает ошибку, если в матрицу признаков попали target-like колонки."""
    leaked = get_target_like_columns(x.columns)
    if leaked:
        raise ValueError(f"Target leakage detected in features: {leaked}")


def assert_no_target_like_columns(x: pd.DataFrame) -> None:
    """Проверяет, что в матрицу признаков не попали endpoint/target-like колонки."""
    leaked = get_target_like_columns(x.columns)
    assert not leaked, f"Feature matrix contains target-like columns: {leaked}"


def build_regression_frame(target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Возвращает X и y для регрессии с anti-leakage правилами."""
    df = load_dataset().copy()
    df = df.dropna(subset=[target_column])
    x = build_feature_matrix(df, target_column=target_column)
    y = df[target_column].astype(float)
    return x, y


def build_binary_target(
    source: pd.Series,
    threshold: float,
    positive_label: str = "greater",
) -> pd.Series:
    """
    Формирует бинарный таргет.

    positive_label оставлен для явного описания в коде, чтобы сигнатура
    функции лучше читалась при создании разных задач.
    """
    _ = positive_label
    return (source > threshold).astype(int)


def split_data(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Единая функция разделения train/test для воспроизводимости."""
    stratify_y = y if stratify else None
    return train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_y,
    )


def make_results_subdir(name: str) -> Path:
    path = results_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))

    baseline = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    residual = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    r2 = 1.0 - residual / baseline if baseline > 0 else 0.0
    return {"RMSE": rmse, "MAE": mae, "R2": r2}
