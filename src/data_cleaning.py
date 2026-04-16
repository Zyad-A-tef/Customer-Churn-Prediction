import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)


def load_data(path_or_url: str = DATA_URL) -> pd.DataFrame:
    """Load the dataset from a local path or a remote URL."""
    df = pd.read_csv(path_or_url)
    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # Missing values
    print("=== Missing Values ===")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "None found")

    # TotalCharges is loaded as object convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    print(f"\nNaN after coercion: {df['TotalCharges'].isna().sum()}")

    df.dropna(subset=["TotalCharges"], inplace=True)
    print(f"Rows after dropping NaN TotalCharges: {len(df):,}")

    df.drop_duplicates(inplace=True)
    df.drop(columns=["customerID"], inplace=True)
    print(f"Final clean shape: {df.shape}")

    #target encoding
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    print("Target distribution:")
    print(df["Churn"].value_counts())

    return df


if __name__ == "__main__":
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    df_clean.to_csv("data/dataset.csv", index=False)
    print("\nSaved cleaned data to data/dataset.csv")
