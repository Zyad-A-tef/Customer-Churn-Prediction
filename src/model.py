import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam

# Machine Learning models

def train_logistic_regression(X_train_sc, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_sc, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def tune_random_forest(X_train, y_train, n_iter: int = 20):
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="f1",
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    print(f"Best CV F1 Score : {search.best_score_:.4f}")
    print(f"Best Parameters  : {search.best_params_}")
    return search.best_estimator_


def train_xgboost(X_train, y_train):
    neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=neg_pos_ratio,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def tune_xgboost(X_train, y_train, n_iter: int = 30):
    neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "scale_pos_weight": [neg_pos_ratio],
    }
    search = RandomizedSearchCV(
        estimator=XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="f1",
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    print(f"Best CV F1 Score : {search.best_score_:.4f}")
    print(f"Best Parameters  : {search.best_params_}")
    return search.best_estimator_

# Deep Learning model (ANN)

def build_ann(input_dim: int, learning_rate: float = 0.001) -> keras.Sequential:
    model = keras.Sequential([
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
    ])

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


def train_ann(X_train_sc, y_train, epochs: int = 150, batch_size: int = 64):
    tf.random.set_seed(42)
    np.random.seed(42)

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train.values)
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

    history = ann.fit(
        X_train_sc, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    return ann, history
