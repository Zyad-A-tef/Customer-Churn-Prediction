from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    df = df.copy()

    binary_cols = [c for c in df.columns
                  if df[c].dtype == 'object' and df[c].nunique() == 2]
    multi_cols  = [c for c in df.columns
                  if df[c].dtype == 'object' and df[c].nunique() > 2]

    df_enc = df.copy()

    le = LabelEncoder()
    for col in binary_cols:
        df_enc[col] = le.fit_transform(df_enc[col])


    df_enc = pd.get_dummies(df_enc, columns=multi_cols, drop_first=True)
    X = df_enc.drop(columns=['Churn'])
    y = df_enc['Churn']
    return X, y