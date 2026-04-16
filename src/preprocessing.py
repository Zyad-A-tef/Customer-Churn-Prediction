import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = df.copy()
    
    # Find all object columns that need encoding
    object_cols = df_enc.select_dtypes(include=['object']).columns.tolist()
    
    binary_cols = [c for c in object_cols if df_enc[c].nunique() == 2]
    multi_cols = [c for c in object_cols if df_enc[c].nunique() > 2]

    print("Binary (Label Encode):", binary_cols)
    print("Multi-value (One-Hot):", multi_cols)

    le = LabelEncoder()
    for col in binary_cols:
        df_enc[col] = le.fit_transform(df_enc[col])

    df_enc = pd.get_dummies(df_enc, columns=multi_cols, drop_first=True)

    print(f"Shape after encoding: {df_enc.shape}")
    print(f"Feature columns: {df_enc.shape[1] - 1}")
    return df_enc


def split_and_scale(
    df_enc: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.20,
    random_state: int = 42,
):

    X = df_enc.drop(columns=[target])
    y = df_enc[target]

    print(f"Features (X): {X.shape}")
    print(f"Target  (y): {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Training set   : {X_train.shape[0]:,} rows ({100*len(X_train)/len(X):.0f}%)")
    print(f"Test set       : {X_test.shape[0]:,} rows ({100*len(X_test)/len(X):.0f}%)")
    print(f"\nChurn rate (train): {y_train.mean():.3f}")
    print(f"Churn rate (test) : {y_test.mean():.3f}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    print("Scaling complete.")

    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler


if __name__ == "__main__":
    from data_cleaning import load_data, clean_data

    df_clean = clean_data(load_data())
    df_enc = encode_features(df_clean)
    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler = split_and_scale(df_enc)
