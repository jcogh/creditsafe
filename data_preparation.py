import pandas as pd
import sqlite3
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from rich.console import Console

console = Console()


def create_sample_data(n_samples=10000):
    np.random.seed(42)
    return pd.DataFrame({
        'income': np.random.lognormal(mean=10.5, sigma=0.5, size=n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'loan_amount': np.random.lognormal(mean=10, sigma=1, size=n_samples),
        'debt': np.random.lognormal(mean=9, sigma=1, size=n_samples),
        'years_employed': np.random.randint(0, 40, n_samples),
        'num_credit_lines': np.random.randint(0, 20, n_samples),
        'is_default': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    })


def load_or_create_data(conn, min_samples=10000):
    try:
        data = pd.read_sql_query("SELECT * FROM credit_data", conn)
        console.print("Existing data loaded from credit_data table.")
        if len(data) < min_samples:
            raise ValueError(
                "Existing dataset is too small. Creating new data.")
    except (pd.io.sql.DatabaseError, ValueError):
        console.print("Creating new sample data...")
        data = create_sample_data(min_samples)
        data.to_sql('credit_data', conn, if_exists='replace', index=False)
        console.print(
            "New sample data created and saved to credit_data table.")
    return data


def engineer_features(data):
    data['debt_to_income'] = data['debt'] / data['income']
    data['loan_to_income'] = data['loan_amount'] / data['income']
    # Assuming credit_score * 10 as max credit
    data['credit_utilization'] = data['debt'] / (data['credit_score'] * 10)
    data['credit_score_bucket'] = pd.qcut(data['credit_score'], q=5, labels=[
                                          'Very Low', 'Low', 'Medium', 'High', 'Very High'])
    return data


def preprocess_data(data):
    # Remove rows with NaN values
    data_clean = data.dropna()
    console.print(
        f"Removed {len(data) - len(data_clean)} rows with NaN values.")

    # Separate features and target
    X = data_clean.drop('is_default', axis=1)
    y = data_clean['is_default']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(
        include=['object', 'category']).columns

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform the data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Get feature names after preprocessing
    onehot_columns = preprocessor.named_transformers_[
        'cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = list(numeric_features) + list(onehot_columns)

    # Convert to DataFrames
    X_train_preprocessed = pd.DataFrame(
        X_train_preprocessed, columns=feature_names)
    X_test_preprocessed = pd.DataFrame(
        X_test_preprocessed, columns=feature_names)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor


def check_data_quality(data):
    console.print("\n[bold]Data Quality Check:[/bold]")
    console.print(f"Total samples: {len(data)}")
    console.print(f"Missing values:\n{data.isnull().sum()}")
    console.print(f"Data types:\n{data.dtypes}")
    console.print(f"Class distribution:\n{
                  data['is_default'].value_counts(normalize=True)}")
    console.print(f"Summary statistics:\n{data.describe()}")


def prepare_data():
    with sqlite3.connect('credit_risk.db') as conn:
        data = load_or_create_data(conn)
        data = engineer_features(data)
        check_data_quality(data)
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data)

        # Save prepared data
        pd.concat([X_train, y_train], axis=1).to_sql(
            'prepared_credit_data_train', conn, if_exists='replace', index=False)
        pd.concat([X_test, y_test], axis=1).to_sql(
            'prepared_credit_data_test', conn, if_exists='replace', index=False)

    console.print(f"Data preparation completed. {len(X_train)} training samples and {
                  len(X_test)} testing samples saved to database.")
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    prepare_data()
