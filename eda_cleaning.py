# === eda_cleaning.py ===
import pandas as pd

def clean_mutual_fund_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    
    # Example column rename if needed
    df.rename(columns={
        'Scheme Name': 'Scheme Name',
        '1 Year Return': '1Y_Return',
        '3 Year Return': '3Y_Return',
        '5 Year Return': '5Y_Return',
        'NAV': 'NAV',
        'Risk': 'Risk',
        'Fund House': 'AMC',
        'Category': 'Category'
    }, inplace=True)
    
    return df
