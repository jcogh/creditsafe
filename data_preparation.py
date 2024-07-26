import pandas as pd
import sqlite3
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def create_sample_data(n_samples=1000):
    np.random.seed(42)
    return pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'loan_amount': np.random.randint(1000, 50000, n_samples),
        'is_default': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })


def load_or_create_data(conn, min_samples=1000):
    try:
        data = pd.read_sql_query("SELECT * FROM credit_data", conn)
        print("Existing data loaded from credit_data table.")
        if len(data) < min_samples:
            raise ValueError(
                "Existing dataset is too small. Creating new data.")
    except (pd.io.sql.DatabaseError, ValueError):
        print("Creating new sample data...")
        data = create_sample_data()
        data.to_sql('credit_data', conn, if_exists='replace', index=False)
        print("New sample data created and saved to credit_data table.")
    return data


def preprocess_data(data):
    data['is_default'] = data['is_default'].astype(int)
    numeric_columns = data.select_dtypes(
        include=['int64', 'float64']).columns.drop('is_default')

    imputer = SimpleImputer(strategy='median')
    data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

    # Feature engineering
    data['credit_utilization'] = data['loan_amount'] / data['income']
    data['age_group'] = pd.cut(data['age'], bins=[0, 30, 45, 60, 100], labels=[
                               '0-30', '31-45', '46-60', '60+'])
    data['age_group'] = data['age_group'].astype(
        'category')  # Ensure it's categorical

    # Log transform skewed features
    data['income'] = np.log1p(data['income'])
    data['loan_amount'] = np.log1p(data['loan_amount'])

    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    return data


def prepare_data():
    with sqlite3.connect('credit_risk.db') as conn:
        data = load_or_create_data(conn)
        prepared_data = preprocess_data(data)
        prepared_data.to_sql('prepared_credit_data', conn,
                             if_exists='replace', index=False)

    print(f"Data preparation completed. {
          len(prepared_data)} samples saved to prepared_credit_data table.")
    print(f"Columns in prepared data: {', '.join(prepared_data.columns)}")
    print(prepared_data.head())
    print(prepared_data.dtypes)


if __name__ == "__main__":
    prepare_data()
