# CreditSafe: A Python-Driven Credit Risk Assessment Model

## Project Overview

CreditSafe is a machine learning project designed to assess credit risk using various predictive models. The project includes data preparation, model development, and evaluation stages, providing insights into credit default prediction.

## Features

- Comprehensive data preparation and preprocessing
- Implementation of multiple machine learning models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- Advanced model evaluation and comparison
- Detailed console output with progress tracking
- SQLite database integration for storing results

## Project Structure

```
creditsafe/
│
├── main.py                 # Main script to run the entire pipeline
├── data_preparation.py     # Script for data loading and preprocessing
├── model_development.py    # Script for model training and evaluation
├── view_database.py        # Script to view the contents of the database
├── requirements.txt        # List of Python dependencies
├── credit_risk.db          # SQLite database for storing data and results
└── README.md               # This file
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/jcogh/creditsafe.git
   cd creditsafe
   ```

2. Create and activate a virtual environment:
   ```
   python3 -m venv creditsafe_env
   source creditsafe_env/bin/activate  # On Windows, use `creditsafe_env\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the entire CreditSafe pipeline:

```
python main.py
```

This will execute the following steps:
1. Data preparation
2. Model development and evaluation
3. Results storage in the SQLite database
4. Display of model performance metrics

To view the contents of the database:

```
python view_database.py
```

## Model Performance

Currently, the models show the following performance:

- **Logistic Regression**: Accuracy 0.7000, AUC 0.5000
- **Random Forest**: Accuracy 0.5600, AUC 0.5018
- **XGBoost**: Accuracy 0.4900, AUC 0.4493
- **LightGBM**: Accuracy (TBD), AUC (TBD)

Note: These results suggest that there is significant room for improvement in the models' predictive power.

## Future Improvements

- **Feature Engineering**: Develop more informative predictors to enhance model accuracy.
- **Address Class Imbalance**: Implement techniques to handle imbalanced datasets more effectively.
- **Hyperparameter Tuning**: Explore a broader range of hyperparameters for optimization.
- **Feature Selection**: Apply techniques to select the most relevant features for modeling.
- **Model Expansion**: Experiment with additional machine learning models to improve predictions.
- **Cross-Validation**: Enhance cross-validation strategies to ensure robust model performance.

## Contributing

Contributions to improve CreditSafe are welcome. Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
