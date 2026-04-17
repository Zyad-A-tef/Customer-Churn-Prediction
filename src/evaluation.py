"""Evaluation helpers shared by the notebook and the CLI runner.

Reports the same metrics used in ``Notebooks/analysis.ipynb``:
Accuracy, F1, ROC-AUC and **PR-AUC** (more informative on imbalanced churn
data), plus confusion matrices, ROC + PR curves, cross-validation summaries,
timed inference, and a tidy comparison table that includes timing.
"""

from __future__ import annotations

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 110


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def print_metrics(name: str, y_test, y_pred, y_proba, fit_time: float | None = None,
                  pred_time: float | None = None) -> None:
    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC   : {average_precision_score(y_test, y_proba):.4f}")
    if fit_time is not None and pred_time is not None:
        print(f"Timing   : train {fit_time:.3f}s | predict {pred_time:.4f}s")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))


def optimal_threshold(y_test, y_proba) -> float:
    """Return the Youden-J optimal decision threshold."""
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    j_scores = tpr - fpr
    best_thresh = float(thresholds[np.argmax(j_scores)])
    print(f"Optimal threshold (Youden): {best_thresh:.3f}")
    return best_thresh


def timed_predict(model, X_test, is_keras: bool = False):
    """Return ``(y_pred, y_proba, pred_time_seconds)``.

    For ``is_keras=True`` callers must apply their own decision threshold
    (``y_pred`` is filled with the 0.5 cut for consistency; replace as needed).
    """
    t0 = perf_counter()
    if is_keras:
        y_proba = np.asarray(model.predict(X_test, verbose=0)).flatten()
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba, perf_counter() - t0


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate_models(models: dict, X_train, y_train, random_state: int = 42) -> pd.DataFrame:
    """Run stratified 5-fold CV for a dict of ``{name: fitted_or_fresh_model}``.

    Uses F1 and ROC-AUC because they are the two metrics that matter most for
    imbalanced churn data; PR-AUC is reported on the hold-out test fold via the
    regular metrics table (it's costlier to compute inside CV loops).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows = []
    for name, model in models.items():
        f1_cv = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        auc_cv = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append(
            {
                "Model": name,
                "F1 (CV mean)": f1_cv.mean(),
                "F1 (CV std)": f1_cv.std(),
                "ROC-AUC (CV mean)": auc_cv.mean(),
                "ROC-AUC (CV std)": auc_cv.std(),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("F1 (CV mean)", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def plot_confusion_matrices(models_info: list, y_test) -> None:
    n = len(models_info)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, pred, _) in zip(axes, models_info):
        cm = confusion_matrix(y_test, pred)
        ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(name, fontweight="bold")
    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_roc_and_pr_curves(models_info: list, y_test,
                           title_prefix: str = "Models") -> None:
    """Plot ROC and Precision-Recall curves side by side."""
    palette = plt.cm.tab10.colors
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5))
    baseline = float(np.mean(y_test))

    for (name, _, proba), color in zip(models_info, palette):
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f"{name} (AUC = {auc(fpr, tpr):.3f})")

        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax_pr.plot(recall, precision, color=color, lw=2,
                   label=f"{name} (AP = {ap:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02)
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curves — {title_prefix}", fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=9)

    ax_pr.axhline(baseline, color="k", linestyle="--", lw=1.2,
                  label=f"Baseline (churn rate = {baseline:.2f})")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-Recall — {title_prefix}", fontweight="bold")
    ax_pr.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(
    results_df: pd.DataFrame, title: str = "Model Comparison"
) -> None:
    metric_cols = [c for c in ["Accuracy", "F1 Score", "ROC-AUC", "PR-AUC"] if c in results_df.columns]
    df = results_df.set_index("Model")[metric_cols]
    ax = df.plot(
        kind="bar", figsize=(12, 5), colormap="tab10", edgecolor="white", rot=15
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7.5, padding=2)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def build_results_table(
    models_info: list,
    y_test,
    model_type_map: dict | None = None,
    timing_log: dict | None = None,
) -> pd.DataFrame:
    """Build a comparison table including PR-AUC and (optional) timing columns.

    ``timing_log`` maps model name -> ``(fit_time, predict_time)``.
    """
    rows = []
    for name, pred, proba in models_info:
        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "F1 Score": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba),
            "PR-AUC": average_precision_score(y_test, proba),
        }
        if timing_log:
            fit_t, pred_t = timing_log.get(name, (np.nan, np.nan))
            row["Train (s)"] = fit_t
            row["Predict (s)"] = pred_t
        if model_type_map:
            row["Type"] = model_type_map.get(name, "")
        rows.append(row)
    return pd.DataFrame(rows)


def print_best_model(results_df: pd.DataFrame) -> None:
    """Print the model with the highest F1 Score in a plain-ASCII banner."""
    best = results_df.loc[results_df["F1 Score"].idxmax()]
    print("\n" + "=" * 60)
    print("BEST MODEL SUMMARY (primary metric: F1 Score)")
    print("=" * 60)
    for col in results_df.columns:
        if col == "Model":
            print(f"  {col:<12}: {best[col]}")
        else:
            val = best[col]
            if isinstance(val, (int, float, np.integer, np.floating)):
                print(f"  {col:<12}: {val:.4f}")
            else:
                print(f"  {col:<12}: {val}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI runner (end-to-end example)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from data_cleaning import clean_data, load_data
    from model import (
        train_ann,
        train_logistic_regression,
        train_random_forest,
        train_xgboost,
        tune_random_forest,
        tune_xgboost,
    )
    from preprocessing import split_and_preprocess

    df_clean = clean_data(load_data())
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_sc,
        X_test_sc,
        preprocessor,
        feature_names,
    ) = split_and_preprocess(df_clean)

    timing: dict[str, tuple[float, float]] = {}

    print("\nTraining Logistic Regression ...")
    lr, lr_fit = train_logistic_regression(X_train_sc, y_train)
    lr_pred, lr_proba, lr_pred_t = timed_predict(lr, X_test_sc)
    timing["Logistic Regression"] = (lr_fit, lr_pred_t)

    print("\nTraining Random Forest (base) ...")
    rf, rf_fit = train_random_forest(X_train, y_train)
    rf_pred, rf_proba, rf_pred_t = timed_predict(rf, X_test)
    timing["Random Forest"] = (rf_fit, rf_pred_t)

    print("\nTuning Random Forest ...")
    rf_tuned, rf_tuned_fit = tune_random_forest(X_train, y_train)
    rf_tuned_pred, rf_tuned_proba, rf_tuned_pred_t = timed_predict(rf_tuned, X_test)
    timing["Random Forest (tuned)"] = (rf_tuned_fit, rf_tuned_pred_t)

    print("\nTraining XGBoost (base) ...")
    xgb, xgb_fit = train_xgboost(X_train, y_train)
    xgb_pred, xgb_proba, xgb_pred_t = timed_predict(xgb, X_test)
    timing["XGBoost"] = (xgb_fit, xgb_pred_t)

    print("\nTuning XGBoost ...")
    xgb_tuned, xgb_tuned_fit = tune_xgboost(X_train, y_train)
    xgb_tuned_pred, xgb_tuned_proba, xgb_tuned_pred_t = timed_predict(xgb_tuned, X_test)
    timing["XGBoost (tuned)"] = (xgb_tuned_fit, xgb_tuned_pred_t)

    print("\nTraining ANN ...")
    ann, history, ann_fit = train_ann(X_train_sc, y_train)
    t0 = perf_counter()
    ann_proba = np.asarray(ann.predict(X_test_sc, verbose=0)).flatten()
    ann_pred_t = perf_counter() - t0
    thresh = optimal_threshold(y_test, ann_proba)
    ann_pred = (ann_proba >= thresh).astype(int)
    timing["ANN"] = (ann_fit, ann_pred_t)

    models_info = [
        ("Logistic Regression", lr_pred, lr_proba),
        ("Random Forest", rf_pred, rf_proba),
        ("Random Forest (tuned)", rf_tuned_pred, rf_tuned_proba),
        ("XGBoost", xgb_pred, xgb_proba),
        ("XGBoost (tuned)", xgb_tuned_pred, xgb_tuned_proba),
        ("ANN", ann_pred, ann_proba),
    ]

    for name, pred, proba in models_info:
        fit_t, pred_t = timing.get(name, (None, None))
        print_metrics(name, y_test, pred, proba, fit_t, pred_t)

    plot_confusion_matrices(models_info, y_test)
    plot_roc_and_pr_curves(models_info, y_test, title_prefix="All Models")

    results_df = build_results_table(
        models_info,
        y_test,
        model_type_map={
            "Logistic Regression": "ML",
            "Random Forest": "ML",
            "Random Forest (tuned)": "ML",
            "XGBoost": "ML",
            "XGBoost (tuned)": "ML",
            "ANN": "DL",
        },
        timing_log=timing,
    )
    print("\n" + results_df.set_index("Model").round(4).to_string())
    plot_metrics_comparison(results_df, title="All Models — Final Performance Comparison")
    print_best_model(results_df)
