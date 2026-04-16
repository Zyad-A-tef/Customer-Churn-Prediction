import pandas as pd
def clean_data(df):
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else 'None found')

    # TotalCharges is loaded as object converted it
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df.dropna()
    df.drop_duplicates(inplace=True)
    df.drop(columns=['customerID'], inplace=True)
    #Encode target col
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    return df