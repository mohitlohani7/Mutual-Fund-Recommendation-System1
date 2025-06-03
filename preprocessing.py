# === preprocessing.py ===
import pandas as pd

def preprocess_data(df):
    # Remove rows with missing NAVs
    df = df.dropna(subset=['NAV'])

    # Convert returns to numeric
    df['1Y_Return'] = pd.to_numeric(df['1Y_Return'], errors='coerce')
    df['3Y_Return'] = pd.to_numeric(df['3Y_Return'], errors='coerce')
    df['5Y_Return'] = pd.to_numeric(df['5Y_Return'], errors='coerce')
    
    df['Risk'] = df['Risk'].fillna('Moderate')
    df = df.dropna(subset=['1Y_Return', '3Y_Return', '5Y_Return'])
    return df

def score_funds(df, risk_profile):
    risk_mapping = {
        'Low': ['Low', 'Moderate'],
        'Moderate': ['Moderate', 'High'],
        'High': ['High']
    }

    filtered = df[df['Risk'].isin(risk_mapping[risk_profile])]

    # Simple scoring based on average returns
    filtered['Score'] = filtered[['1Y_Return', '3Y_Return', '5Y_Return']].mean(axis=1)
    return filtered.sort_values(by='Score', ascending=False)
