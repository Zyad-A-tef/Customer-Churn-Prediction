import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 110



def print_metrics(name: str, y_test, y_pred, y_proba):
    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))


def optimal_threshold(y_test, y_proba) -> float:
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    j_scores = tpr - fpr
    best_thresh = thresholds[np.argmax(j_scores)]
    print(f"Optimal threshold (Youden): {best_thresh:.3f}")
    return best_thresh


# Visualisations
def plot_confusion_matrices(models_info: list, y_test):
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


def plot_roc_curves(models_info: list, y_test, title: str = "ROC Curves"):
    palette = plt.cm.tab10.colors
    plt.figure(figsize=(8, 6))
    for (name, _, proba), color in zip(models_info, palette):
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(results_df: pd.DataFrame, title: str = "Model Comparison"):
    df = results_df.set_index("Model")[["Accuracy", "F1 Score", "ROC-AUC"]]
    ax = df.plot(kind="bar", figsize=(12, 5), colormap="tab10",
                 edgecolor="white", rot=15)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7.5, padding=2)
    plt.tight_layout()
    plt.show()


def build_results_table(models_info: list, y_test, model_type_map: dict = None) -> pd.DataFrame:
    rows = []
    for name, pred, proba in models_info:
        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "F1 Score": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba),
        }
        if model_type_map:
            row["Type"] = model_type_map.get(name, "")
        rows.append(row)
    return pd.DataFrame(rows)


def print_best_model(results_df: pd.DataFrame):
    """Print the model with the highest F1 Score."""
    best = results_df.loc[results_df["F1 Score"].idxmax()]
    print("\n" + "🏆 " * 20)
    print("   BEST MODEL SUMMARY")
    print("🏆 " * 20)
    for col in ["Model", "Accuracy", "F1 Score", "ROC-AUC"]:
        val = best[col]
        print(f"  {col:<10}: {val:.4f}" if isinstance(val, float) else f"  {col:<10}: {val}")


if __name__ == "__main__":
    from data_cleaning import load_data, clean_data
    from preprocessing import encode_features, split_and_scale
    from model import (
        train_logistic_regression,
        train_random_forest, tune_random_forest,
        train_xgboost, tune_xgboost,
        train_ann
    )
    import numpy as np

    df = clean_data(load_data())
    df_enc = encode_features(df)
    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler = split_and_scale(df_enc)

    # --- Train all models ---
    print("\nTraining Logistic Regression...")
    lr = train_logistic_regression(X_train_sc, y_train)

    print("\nTraining Random Forest (base)...")
    rf = train_random_forest(X_train, y_train)

    print("\nTuning Random Forest...")
    rf_tuned = tune_random_forest(X_train, y_train)

    print("\nTraining XGBoost (base)...")
    xgb = train_xgboost(X_train, y_train)

    print("\nTuning XGBoost...")
    xgb_tuned = tune_xgboost(X_train, y_train)

    print("\nTraining ANN...")
    ann, _ = train_ann(X_train_sc, y_train)

    # --- Predictions ---
    ann_proba_test = ann.predict(X_test_sc, verbose=0).flatten()
    thresh = optimal_threshold(y_test, ann_proba_test)
    ann_pred = (ann_proba_test >= thresh).astype(int)

    models_info = [
        ("Logistic Regression",  lr.predict(X_test_sc),       lr.predict_proba(X_test_sc)[:, 1]),
        ("Random Forest",        rf.predict(X_test),           rf.predict_proba(X_test)[:, 1]),
        ("Random Forest (tuned)", rf_tuned.predict(X_test),   rf_tuned.predict_proba(X_test)[:, 1]),
        ("XGBoost",              xgb.predict(X_test),          xgb.predict_proba(X_test)[:, 1]),
        ("XGBoost (tuned)",      xgb_tuned.predict(X_test),   xgb_tuned.predict_proba(X_test)[:, 1]),
        ("ANN",                  ann_pred,                      ann_proba_test),
    ]

    # --- Print metrics for each ---
    for name, pred, proba in models_info:
        print_metrics(name, y_test, pred, proba)

    # --- Visualisations ---
    plot_confusion_matrices(models_info, y_test)
    plot_roc_curves(models_info, y_test, title="ROC Curves — All Models")

    results_df = build_results_table(
        models_info, y_test,
        model_type_map={
            "Logistic Regression": "ML",
            "Random Forest": "ML",
            "Random Forest (tuned)": "ML",
            "XGBoost": "ML",
            "XGBoost (tuned)": "ML",
            "ANN": "DL",
        }
    )
    print("\n", results_df.set_index("Model").round(4).to_string())
    plot_metrics_comparison(results_df, title="All Models — Final Performance Comparison")
    print_best_model(results_df)