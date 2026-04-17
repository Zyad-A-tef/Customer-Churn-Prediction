"""Data loading and cleaning for the Telco Customer Churn dataset.

This module mirrors the cleaning step of ``Notebooks/analysis.ipynb``. It drops
target-leakage columns (``Churn Label``/``Churn Score``/``Churn Reason``/``Count``)
and high-cardinality geographic columns (``Country``/``State``/``City``/
``Zip Code``/``Lat Long``/``Latitude``/``Longitude``) which add no predictive
signal but explode the feature space after one-hot encoding.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Telco_customer_churn.xlsx"
FALLBACK_KAGGLE_DATASET = "yeanzc/telco-customer-churn-ibm-dataset"
FALLBACK_KAGGLE_FILE_PATH = ""

COLUMN_RENAME_MAP = {
    "CustomerID": "customerID",
    "Gender": "gender",
    "Senior Citizen": "SeniorCitizen",
    "Tenure Months": "tenure",
    "Phone Service": "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup": "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
}

LEAKAGE_COLUMNS = ["Churn Label", "Churn Score", "Churn Reason", "Count"]

# Geographic / non-predictive columns. All customers live in California, so
# Country / State are constant; City, Zip Code and Lat Long are near-unique and
# would create thousands of one-hot dummies without adding signal.
GEO_COLUMNS = [
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
]


def load_data(path_or_url: str | None = None) -> pd.DataFrame:
    """Load dataset from xlsx/csv (local) or URL fallback."""
    source = path_or_url or str(DATA_PATH)
    if source.endswith((".xlsx", ".xls")):
        df = pd.read_excel(source)
    elif source.endswith(".csv") or source.startswith(("http://", "https://")):
        df = pd.read_csv(source)
    else:
        raise ValueError(f"Unsupported file format for source: {source}")

    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def load_data_from_kagglehub(
    dataset: str = FALLBACK_KAGGLE_DATASET,
    file_path: str = FALLBACK_KAGGLE_FILE_PATH,
) -> pd.DataFrame:
    """Load dataset from KaggleHub using pandas adapter."""
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError as exc:
        raise ImportError(
            "kagglehub is not installed. Run: pip install kagglehub[pandas-datasets]"
        ) from exc

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        dataset,
        file_path,
    )
    print(f"Loaded dataset from KaggleHub: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Telco dataset and return a modeling-ready DataFrame.

    Returns a frame with a single integer ``Churn`` target column and all
    predictor columns in pipeline-ready names.
    """
    df = df.copy()
    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    print("=== Missing Values ===")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "None found")

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        print(f"\nNaN after TotalCharges coercion: {df['TotalCharges'].isna().sum()}")
        df.dropna(subset=["TotalCharges"], inplace=True)
        print(f"Rows after dropping NaN TotalCharges: {len(df):,}")

    df.drop_duplicates(inplace=True)

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Target harmonization (do this BEFORE dropping leakage so we can still read
    # Churn Label / Churn Value whichever schema is present).
    if "Churn Value" in df.columns:
        df["Churn"] = df["Churn Value"].astype(int)
        df.drop(columns=["Churn Value"], inplace=True)
    elif "Churn Label" in df.columns and "Churn" not in df.columns:
        df["Churn"] = (df["Churn Label"] == "Yes").astype(int)
    elif "Churn" in df.columns:
        if df["Churn"].dtype == object:
            df["Churn"] = (df["Churn"] == "Yes").astype(int)
        else:
            df["Churn"] = df["Churn"].astype(int)
    else:
        raise KeyError(
            "No churn target column found. Expected one of: "
            "['Churn', 'Churn Value', 'Churn Label']"
        )

    leakage_dropped = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if leakage_dropped:
        df.drop(columns=leakage_dropped, inplace=True)
        print(f"Dropped leakage columns: {leakage_dropped}")

    geo_dropped = [c for c in GEO_COLUMNS if c in df.columns]
    if geo_dropped:
        df.drop(columns=geo_dropped, inplace=True)
        print(f"Dropped geographic columns: {geo_dropped}")

    print(f"Final clean shape: {df.shape}")
    print("Target distribution:")
    print(df["Churn"].value_counts())

    return df


if __name__ == "__main__":
    if DATA_PATH.exists():
        df_raw = load_data(str(DATA_PATH))
    else:
        print(f"Local dataset not found at {DATA_PATH}, using KaggleHub fallback.")
        df_raw = load_data_from_kagglehub()
    df_clean = clean_data(df_raw)
    output_path = Path(__file__).resolve().parents[1] / "data" / "dataset.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\nSaved cleaned data to {output_path}")
