import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()


def load_data(db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    console.print(
        f"[bold]Columns in loaded data:[/bold] {', '.join(data.columns)}")
    console.print(data.head().to_string())
    console.print(data.dtypes)
    return data


def split_data(data):
    X = data.drop('is_default', axis=1)
    y = data['is_default']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }


def create_pipeline(estimator, numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first',
             sparse_output=False), categorical_features)
        ])

    return ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', estimator)
    ])


def train_and_evaluate_models():
    data = load_data('credit_risk.db', 'prepared_credit_data')
    X_train, X_test, y_train, y_test = split_data(data)

    numeric_features = X_train.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=['category', 'object']).columns.tolist()

    console.print(
        f"[bold]Numeric features:[/bold] {', '.join(numeric_features)}")
    console.print(
        f"[bold]Categorical features:[/bold] {', '.join(categorical_features)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression
    lr_pipeline = create_pipeline(LogisticRegression(
        random_state=42, solver='liblinear'), numeric_features, categorical_features)
    lr_params = {
        'classifier__C': np.logspace(-4, 4, 20), 'classifier__penalty': ['l1', 'l2']}
    lr_model = RandomizedSearchCV(
        lr_pipeline, lr_params, n_iter=20, cv=3, random_state=42)

    # Random Forest
    rf_pipeline = create_pipeline(RandomForestClassifier(
        random_state=42), numeric_features, categorical_features)
    rf_params = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    rf_model = RandomizedSearchCV(
        rf_pipeline, rf_params, n_iter=20, cv=3, random_state=42)

    # XGBoost
    xgb_pipeline = create_pipeline(XGBClassifier(
        random_state=42, enable_categorical=True), numeric_features, categorical_features)
    xgb_params = {
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.3],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__subsample': [0.6, 0.8, 1.0]
    }
    xgb_model = RandomizedSearchCV(
        xgb_pipeline, xgb_params, n_iter=20, cv=3, random_state=42)

    models = [
        ('Logistic Regression', lr_model),
        ('Random Forest', rf_model),
        ('XGBoost', xgb_model)
    ]

    results = []
    for name, model in tqdm(models, desc="Training models"):
        console.print(f"[bold]Training and evaluating {name}...[/bold]")
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring='roc_auc')
        evaluation = evaluate_model(model, X_test, y_test)
        results.append((name, cv_scores, evaluation))
        console.print(f"[green]{name} evaluation:[/green] {evaluation}")

    return results


def store_results(results, db_path):
    results_df = pd.DataFrame([
        {
            'model': name,
            'cv_score': np.mean(cv_scores),
            'accuracy': evaluation['accuracy'],
            'precision': evaluation['precision'],
            'recall': evaluation['recall'],
            'f1_score': evaluation['f1_score'],
            'auc': evaluation['auc']
        }
        for name, cv_scores, evaluation in results
    ])

    with sqlite3.connect(db_path) as conn:
        results_df.to_sql('model_results', conn,
                          if_exists='replace', index=False)


if __name__ == "__main__":
    results = train_and_evaluate_models()
    store_results(results, 'credit_risk.db')
    console.print(
        "[bold green]Model development and evaluation completed. Results stored in the database.[/bold green]")
