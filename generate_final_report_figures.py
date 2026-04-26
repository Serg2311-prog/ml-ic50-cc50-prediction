"""Generate final report figures with descriptor-only anti-leakage pipelines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_common import (
    RANDOM_STATE,
    data_path,
    get_target_like_columns,
    load_dataset,
    validate_no_target_like_columns,
)


OUTPUT_DIR = Path("artifacts") / "final_report_figures"
TARGETS = ["IC50, mM", "CC50, mM", "SI"]


def descriptor_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptor-only X and validate that no target-like columns remain."""
    target_like = set(get_target_like_columns(df.columns))
    x = df[[col for col in df.columns if col not in target_like]].copy()
    validate_no_target_like_columns(x)
    return x


def save_current_figure(filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close()


def plot_log1p_comparison(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for row, target in enumerate(TARGETS):
        sns.histplot(df[target].dropna(), kde=True, bins=40, ax=axes[row, 0], color="#4C72B0")
        axes[row, 0].set_title(f"{target}: before log1p")
        axes[row, 0].set_xlabel(target)

        sns.histplot(np.log1p(df[target].dropna()), kde=True, bins=40, ax=axes[row, 1], color="#55A868")
        axes[row, 1].set_title(f"{target}: after log1p")
        axes[row, 1].set_xlabel(f"log1p({target})")

    fig.suptitle("Target distributions before and after log1p transformation", y=1.02)
    save_current_figure("01_before_after_log1p_targets.png")


def fit_regression_pipeline(
    df: pd.DataFrame,
    target: str,
    model_name: str,
    model,
) -> tuple[pd.Series, np.ndarray, Pipeline, pd.Index]:
    data = df.dropna(subset=[target]).copy()
    x = descriptor_matrix(data)
    y = data[target].astype(float)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    return y_test, y_pred, pipeline, x.columns


def plot_fact_vs_prediction(df: pd.DataFrame) -> dict[str, tuple[Pipeline, pd.Index]]:
    regression_models = {
        "IC50, mM": (
            "Gradient Boosting",
            GradientBoostingRegressor(random_state=RANDOM_STATE),
        ),
        "CC50, mM": (
            "Random Forest",
            RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1),
        ),
        "SI": (
            "Gradient Boosting",
            GradientBoostingRegressor(random_state=RANDOM_STATE),
        ),
    }

    fitted: dict[str, tuple[Pipeline, pd.Index]] = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (target, (model_name, model)) in zip(axes, regression_models.items()):
        y_test, y_pred, pipeline, feature_names = fit_regression_pipeline(df, target, model_name, model)
        fitted[target] = (pipeline, feature_names)

        ax.scatter(y_test, y_pred, alpha=0.6, edgecolor="none")
        low = min(y_test.min(), y_pred.min())
        high = max(y_test.max(), y_pred.max())
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{target}: {model_name}")
        ax.set_xlabel("Fact / y_true")
        ax.set_ylabel("Prediction / y_pred")

    fig.suptitle("Fact vs prediction for best regression models", y=1.03)
    save_current_figure("02_fact_vs_prediction_regression.png")
    return fitted


def plot_feature_importance(df: pd.DataFrame) -> None:
    importance_targets = ["IC50, mM", "CC50, mM"]
    fig, axes = plt.subplots(len(importance_targets), 2, figsize=(16, 10))

    for row, target in enumerate(importance_targets):
        data = df.dropna(subset=[target]).copy()
        x = descriptor_matrix(data)
        y = data[target].astype(float)
        x_train, _, y_train, _ = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
        models = [
            (
                "Random Forest",
                RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1),
            ),
            (
                "Gradient Boosting",
                GradientBoostingRegressor(random_state=RANDOM_STATE),
            ),
        ]
        for col_idx, (model_name, model) in enumerate(models):
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", model),
                ]
            )
            pipeline.fit(x_train, y_train)
            importances = pd.Series(
                pipeline.named_steps["model"].feature_importances_,
                index=x.columns,
            ).sort_values(ascending=False).head(15)
            sns.barplot(
                x=importances.values,
                y=importances.index,
                ax=axes[row, col_idx],
                color="#4C72B0",
            )
            axes[row, col_idx].set_title(f"{target}: {model_name}")
            axes[row, col_idx].set_xlabel("Feature importance")
            axes[row, col_idx].set_ylabel("Descriptor")

    fig.suptitle("Top descriptor importances for IC50 and CC50 models", y=1.02)
    save_current_figure("03_feature_importance_ic50_cc50.png")


def fit_classification_pipeline(df: pd.DataFrame, target_col: str, threshold: float) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    data = df.dropna(subset=[target_col]).copy()
    x = descriptor_matrix(data)
    y = (data[target_col] > threshold).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]
    return y_test, y_pred, y_proba


def classification_tasks(df: pd.DataFrame) -> dict[str, tuple[str, float]]:
    return {
        "IC50 > median": ("IC50, mM", float(df["IC50, mM"].median())),
        "CC50 > median": ("CC50, mM", float(df["CC50, mM"].median())),
        "SI > median": ("SI", float(df["SI"].median())),
        "SI > 8": ("SI", 8.0),
    }


def plot_confusion_matrices(df: pd.DataFrame) -> dict[str, tuple[pd.Series, np.ndarray, np.ndarray]]:
    diagnostics = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.reshape(-1)
    for ax, (task_name, (target_col, threshold)) in zip(axes, classification_tasks(df).items()):
        y_test, y_pred, y_proba = fit_classification_pipeline(df, target_col, threshold)
        diagnostics[task_name] = (y_test, y_pred, y_proba)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=False)
        ax.set_title(task_name)

    fig.suptitle("Confusion matrices for best classification models", y=1.02)
    save_current_figure("04_confusion_matrices_classification.png")
    return diagnostics


def plot_roc_and_pr_curves(diagnostics: dict[str, tuple[pd.Series, np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    for idx, (task_name, (y_test, _, y_proba)) in enumerate(diagnostics.items()):
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0, idx])
        axes[0, idx].set_title(f"ROC: {task_name}")

        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[1, idx])
        axes[1, idx].set_title(f"PR: {task_name}")

    fig.suptitle("ROC and precision-recall curves for best classification models", y=1.02)
    save_current_figure("05_roc_curves_and_06_pr_curves_classification.png")


def main() -> None:
    if not data_path().exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path()}. "
            "Place the Excel file there before generating final report figures."
        )

    sns.set(style="whitegrid", context="notebook")
    df = load_dataset()

    plot_log1p_comparison(df)
    plot_fact_vs_prediction(df)
    plot_feature_importance(df)
    classification_diagnostics = plot_confusion_matrices(df)
    plot_roc_and_pr_curves(classification_diagnostics)

    print(f"Saved final report figures to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
