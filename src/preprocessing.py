"""Preprocessing pipeline for the Telco Customer Churn dataset.

Uses a single ``ColumnTransformer`` fitted on the training fold only so no
statistics (scaler mean/std, encoder categories) leak from test into train:

- ``StandardScaler`` on continuous numerics (``tenure``, ``MonthlyCharges``,
  ``TotalCharges``, ``CLTV``). Scaling one-hot dummies would distort
  interpretation without improving any model.
- ``OneHotEncoder(drop='if_binary', handle_unknown='ignore')`` on categoricals.
- ``remainder='passthrough'`` keeps already-numeric 0/1 fields as-is.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CONTINUOUS_CANDIDATES = ("tenure", "MonthlyCharges", "TotalCharges", "CLTV")


def identify_feature_types(
    df: pd.DataFrame, target: str = "Churn"
) -> tuple[list[str], list[str], list[str]]:
    """Return (continuous_cols, categorical_cols, passthrough_cols) for ``df``."""
    feature_df = df.drop(columns=[target]) if target in df.columns else df

    continuous_cols = [c for c in CONTINUOUS_CANDIDATES if c in feature_df.columns]
    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()
    passthrough_cols = [
        c
        for c in feature_df.columns
        if c not in continuous_cols + categorical_cols
    ]
    return continuous_cols, categorical_cols, passthrough_cols


def build_preprocessor(continuous_cols, categorical_cols) -> ColumnTransformer:
    """Build the shared ColumnTransformer."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), continuous_cols),
            (
                "cat",
                OneHotEncoder(
                    drop="if_binary", handle_unknown="ignore", sparse_output=False
                ),
                categorical_cols,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def split_and_preprocess(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Split ``df`` into train/test folds and fit the preprocessor on train only.

    Returns
    -------
    X_train, X_test : pandas.DataFrame
        Encoded + scaled design matrices with feature names as column labels.
    y_train, y_test : pandas.Series
    X_train_sc, X_test_sc : numpy.ndarray
        Same arrays as DataFrames above; provided for scale-sensitive models
        such as Logistic Regression and the ANN.
    preprocessor : ColumnTransformer
        Fitted transformer (reusable at inference time).
    feature_names : list[str]
        Final column names after transformation.
    """
    X_raw = df.drop(columns=[target])
    y = df[target].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    continuous_cols, categorical_cols, passthrough_cols = identify_feature_types(df, target)
    preprocessor = build_preprocessor(continuous_cols, categorical_cols)

    X_train_arr = preprocessor.fit_transform(X_train_raw)
    X_test_arr = preprocessor.transform(X_test_raw)
    feature_names = list(preprocessor.get_feature_names_out())

    X_train = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train_raw.index)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test_raw.index)

    print(f"Features (X)  : {X_raw.shape[1]} raw -> {len(feature_names)} encoded")
    print(f"Training set  : {X_train.shape[0]:,} rows ({100 * len(X_train) / len(X_raw):.0f}%)")
    print(f"Test set      : {X_test.shape[0]:,} rows ({100 * len(X_test) / len(X_raw):.0f}%)")
    print(f"Churn (train) : {y_train.mean():.3f} | Churn (test) : {y_test.mean():.3f}")
    print(f"Continuous scaled : {len(continuous_cols)} | Categorical OHE : {len(categorical_cols)} | Passthrough : {len(passthrough_cols)}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        np.asarray(X_train_arr),
        np.asarray(X_test_arr),
        preprocessor,
        feature_names,
    )


if __name__ == "__main__":
    from data_cleaning import clean_data, load_data

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
