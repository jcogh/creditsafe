from data_preparation import prepare_data
from model_development import train_and_evaluate_models, store_results

if __name__ == "__main__":
    print("Starting CreditSafe: A Python-Driven Credit Risk Assessment Model")
    
    print("\nStep 1: Data Preparation")
    prepare_data()
    
    print("\nStep 2: Model Development and Evaluation")
    lr_model, rf_model, lr_cv_scores, rf_cv_scores, lr_evaluation, rf_evaluation = train_and_evaluate_models()
    
    print("\nStep 3: Storing Results")
    store_results(lr_cv_scores, rf_cv_scores, lr_evaluation, rf_evaluation)
    
    print("\nCreditSafe project completed successfully!")
