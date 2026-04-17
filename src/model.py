"""Model builders for the Customer Churn Prediction project.

Three ML baselines (Logistic Regression, Random Forest, XGBoost) plus a small
feed-forward ANN. Each trainer returns a fitted estimator plus the measured
training time so the downstream ML-vs-DL comparison can report latency
alongside quality metrics.
"""

from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers
from tensorflow.keras.optimizers import Adam


def _timed_fit(model, X_train, y_train):
    """Fit a sklearn-style model and return (model, fit_time_seconds)."""
    t0 = perf_counter()
    model.fit(X_train, y_train)
    return model, perf_counter() - t0


# ---------------------------------------------------------------------------
# Machine learning models
# ---------------------------------------------------------------------------


def train_logistic_regression(X_train_sc, y_train, random_state: int = 42):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )
    return _timed_fit(model, X_train_sc, y_train)


def train_random_forest(X_train, y_train, random_state: int = 42):
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return _timed_fit(model, X_train, y_train)


def tune_random_forest(X_train, y_train, n_iter: int = 20, random_state: int = 42):
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=random_state),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="f1",
        verbose=1,
        n_jobs=-1,
        random_state=random_state,
    )
    search.fit(X_train, y_train)
    print(f"Best CV F1 Score : {search.best_score_:.4f}")
    print(f"Best Parameters  : {search.best_params_}")

    best = RandomForestClassifier(
        **search.best_params_, random_state=random_state, n_jobs=-1
    )
    return _timed_fit(best, X_train, y_train)


def _neg_pos_ratio(y_train) -> float:
    return float((np.asarray(y_train) == 0).sum() / max((np.asarray(y_train) == 1).sum(), 1))


def train_xgboost(X_train, y_train, random_state: int = 42):
    neg_pos = _neg_pos_ratio(y_train)
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=neg_pos,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    return _timed_fit(model, X_train, y_train)


def tune_xgboost(X_train, y_train, n_iter: int = 25, random_state: int = 42):
    neg_pos = _neg_pos_ratio(y_train)
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    search = RandomizedSearchCV(
        estimator=XGBClassifier(
            scale_pos_weight=neg_pos,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=5,
        scoring="f1",
        verbose=0,
        n_jobs=-1,
        random_state=random_state,
    )
    search.fit(X_train, y_train)
    print(f"Best CV F1 Score : {search.best_score_:.4f}")
    print(f"Best Parameters  : {search.best_params_}")

    best = XGBClassifier(
        **search.best_params_,
        scale_pos_weight=neg_pos,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    return _timed_fit(best, X_train, y_train)


# ---------------------------------------------------------------------------
# Deep learning model (ANN)
# ---------------------------------------------------------------------------


def build_ann(input_dim: int, learning_rate: float = 0.001) -> keras.Sequential:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def train_ann(
    X_train_sc,
    y_train,
    epochs: int = 150,
    batch_size: int = 64,
    random_state: int = 42,
):
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    y_arr = np.asarray(y_train)
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_arr)
    class_weights = {0: cw[0], 1: cw[1]}
    print(f"Class weights: {class_weights}")

    ann = build_ann(X_train_sc.shape[1])
    ann.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_auc", patience=15, restore_best_weights=True, mode="max"
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=0
    )

    t0 = perf_counter()
    history = ann.fit(
        X_train_sc,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    fit_time = perf_counter() - t0
    print(f"ANN training time: {fit_time:.2f}s over {len(history.history['loss'])} epochs")
    return ann, history, fit_time
