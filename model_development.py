import pandas as pd
import sqlite3
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
import warnings

console = Console()


def load_data(db_path):
    with sqlite3.connect(db_path) as conn:
        X_train = pd.read_sql_query(
            "SELECT * FROM prepared_credit_data_train", conn)
        X_test = pd.read_sql_query(
            "SELECT * FROM prepared_credit_data_test", conn)

    y_train = X_train.pop('is_default')
    y_test = X_test.pop('is_default')

    # Check for NaN values
    if X_train.isnull().any().any() or X_test.isnull().any().any() or y_train.isnull().any() or y_test.isnull().any():
        console.print(
            "[bold red]Warning: NaN values detected in the data. Please check the data preparation step.[/bold red]")
        console.print(f"NaN in X_train: {X_train.isnull().sum().sum()}")
        console.print(f"NaN in X_test: {X_test.isnull().sum().sum()}")
        console.print(f"NaN in y_train: {y_train.isnull().sum()}")
        console.print(f"NaN in y_test: {y_test.isnull().sum()}")
        raise ValueError(
            "NaN values found in the data. Please handle these before proceeding.")

    return X_train, X_test, y_train, y_test


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


def create_pipeline(estimator):
    return ImbPipeline([
        ('feature_selection', SelectFromModel(
            estimator=RandomForestClassifier(n_estimators=100, random_state=42))),
        ('smote', SMOTE(random_state=42)),
        ('classifier', estimator)
    ])


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = [
        ('Logistic Regression', LogisticRegression(random_state=42, solver='liblinear'), {
            'classifier__C': np.logspace(-4, 4, 20),
            'classifier__penalty': ['l1', 'l2'],
            'classifier__class_weight': ['balanced', None]
        }),
        ('Random Forest', RandomForestClassifier(random_state=42), {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__class_weight': ['balanced', 'balanced_subsample', None]
        }),
        ('XGBoost', XGBClassifier(random_state=42), {
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.3],
            'classifier__n_estimators': [100, 200, 300],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__scale_pos_weight': [1, 5, 10]
        }),
        ('LightGBM', LGBMClassifier(random_state=42, verbose=-1, silent=True), {
            'classifier__num_leaves': [31, 63, 127],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.3],
            'classifier__n_estimators': [100, 200, 300],
            'classifier__min_child_samples': [5, 10, 20],
            'classifier__scale_pos_weight': [1, 5, 10]
        })
    ]

    results = []
    for name, estimator, param_grid in tqdm(models, desc="Training models"):
        console.print(f"[bold]Training and evaluating {name}...[/bold]")
        pipeline = create_pipeline(estimator)
        model = RandomizedSearchCV(
            pipeline, param_grid, n_iter=50, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model.fit(X_train, y_train)
        evaluation = evaluate_model(model, X_test, y_test)
        results.append((name, model, model.best_score_, evaluation))
        console.print(f"[green]{name} evaluation:[/green] {evaluation}")

    return results


def store_results(results, db_path):
    results_df = pd.DataFrame([
        {
            'model': name,
            'cv_score': cv_score,
            'accuracy': evaluation['accuracy'],
            'precision': evaluation['precision'],
            'recall': evaluation['recall'],
            'f1_score': evaluation['f1_score'],
            'auc': evaluation['auc']
        }
        for name, model, cv_score, evaluation in results
    ])

    with sqlite3.connect(db_path) as conn:
        results_df.to_sql('model_results', conn,
                          if_exists='replace', index=False)


def display_feature_importance(model, feature_names):
    console.print("[bold]Attempting to display feature importance...[/bold]")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        console.print("Using feature_importances_ attribute")
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        console.print("Using coef_ attribute")
    else:
        console.print(
            "[red]Feature importance not available for this model.[/red]")
        return

    console.print(f"Number of importance values: {len(importances)}")
    console.print(f"Number of feature names: {len(feature_names)}")

    # Ensure importances and feature_names have the same length
    min_length = min(len(importances), len(feature_names))
    importances = importances[:min_length]
    feature_names = feature_names[:min_length]

    feature_importance = pd.DataFrame(
        {'feature': feature_names, 'importance': importances})
    feature_importance = feature_importance.sort_values(
        'importance', ascending=False).head(10)

    table = Table(title="Top 10 Feature Importances")
    table.add_column("Feature", style="cyan")
    table.add_column("Importance", style="magenta")

    for _, row in feature_importance.iterrows():
        table.add_row(row['feature'], f"{row['importance']:.4f}")

    console.print(table)


if __name__ == "__main__":
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            X_train, X_test, y_train, y_test = load_data('credit_risk.db')
            results = train_and_evaluate_models(
                X_train, X_test, y_train, y_test)
            store_results(results, 'credit_risk.db')
            console.print(
                "[bold green]Model development and evaluation completed. Results stored in the database.[/bold green]")

            # Display feature importance for the best model
            best_model = max(results, key=lambda x: x[2]['auc'])[1]
            display_feature_importance(
                best_model.best_estimator_.named_steps['classifier'], X_train.columns)
    except Exception as e:
        console.print(f"[bold red]An error occurred: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
