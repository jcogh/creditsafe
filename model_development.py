import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def load_data():
    conn = sqlite3.connect('credit_risk.db')
    data = pd.read_sql_query("SELECT * FROM prepared_credit_data", conn)
    conn.close()
    return data

def split_data(data):
    X = data.drop('is_default', axis=1)
    y = data['is_default']
    if len(np.unique(y)) < 2:
        raise ValueError("Dataset does not contain samples from all classes.")
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

def train_and_evaluate_models():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_cv_scores = cross_val_score(lr_model, X_train, y_train, cv=cv, scoring='roc_auc')
    lr_evaluation = evaluate_model(lr_model, X_test, y_test)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='roc_auc')
    rf_evaluation = evaluate_model(rf_model, X_test, y_test)

    return lr_model, rf_model, lr_cv_scores, rf_cv_scores, lr_evaluation, rf_evaluation

def store_results(lr_cv_scores, rf_cv_scores, lr_evaluation, rf_evaluation):
    results_df = pd.DataFrame({
        'model': ['Logistic Regression', 'Random Forest'],
        'cv_score': [np.mean(lr_cv_scores), np.mean(rf_cv_scores)],
        'accuracy': [lr_evaluation['accuracy'], rf_evaluation['accuracy']],
        'precision': [lr_evaluation['precision'], rf_evaluation['precision']],
        'recall': [lr_evaluation['recall'], rf_evaluation['recall']],
        'f1_score': [lr_evaluation['f1_score'], rf_evaluation['f1_score']],
        'auc': [lr_evaluation['auc'], rf_evaluation['auc']]
    })

    conn = sqlite3.connect('credit_risk.db')
    results_df.to_sql('model_results', conn, if_exists='replace', index=False)
    conn.close()

if __name__ == "__main__":
    lr_model, rf_model, lr_cv_scores, rf_cv_scores, lr_evaluation, rf_evaluation = train_and_evaluate_models()
    store_results(lr_cv_scores, rf_cv_scores, lr_evaluation, rf_evaluation)
    print("Model development and evaluation completed. Results stored in the database.")
