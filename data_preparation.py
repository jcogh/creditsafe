import pandas as pd
import sqlite3
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def prepare_data():
    conn = sqlite3.connect('credit_risk.db')

    try:
        data = pd.read_sql_query("SELECT * FROM credit_data", conn)
        print("Existing data loaded from credit_data table.")
        if len(data) < 100:  # If the existing data is too small, create new data
            raise ValueError("Existing dataset is too small. Creating new data.")
    except (pd.io.sql.DatabaseError, ValueError):
        print("Creating new sample data...")
        np.random.seed(42)
        n_samples = 1000
        data = pd.DataFrame({
            'age': np.random.randint(18, 70, n_samples),
            'income': np.random.randint(20000, 150000, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'loan_amount': np.random.randint(1000, 50000, n_samples),
            'is_default': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        })
        
        data.to_sql('credit_data', conn, if_exists='replace', index=False)
        print("New sample data created and saved to credit_data table.")

    data['is_default'] = data['is_default'].astype(int)

    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.drop('is_default')
    
    imputer = SimpleImputer(strategy='mean')
    data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
    
    if 'credit_limit' in data.columns and 'balance' in data.columns:
        data['credit_utilization'] = data['balance'] / data['credit_limit']
    
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    data.to_sql('prepared_credit_data', conn, if_exists='replace', index=False)
    
    conn.close()
    print(f"Data preparation completed. {len(data)} samples saved to prepared_credit_data table.")

if __name__ == "__main__":
    prepare_data()
