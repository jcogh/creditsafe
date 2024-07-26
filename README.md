# CreditSafe: A Python-Driven Credit Risk Assessment Model

## Project Overview

CreditSafe is a machine learning project designed to assess credit risk using various predictive models. The project includes data preparation, model development, and evaluation stages, providing insights into credit default prediction.

## Features

- Data preparation and preprocessing
- Implementation of multiple machine learning models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Model evaluation and comparison
- Rich console output with progress bars

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
3. Results storage
4. Display of model performance metrics

To view the contents of the database:

```
python view_database.py
```

## Model Performance

Currently, the models show the following performance:

- Logistic Regression: Accuracy 0.7000, AUC 0.5000
- Random Forest: Accuracy 0.5600, AUC 0.5018
- XGBoost: Accuracy 0.4900, AUC 0.4493

Note: These results indicate that there's room for improvement in the models' predictive power.

## Future Improvements

- Feature engineering to create more informative predictors
- Address class imbalance issues
- Expand hyperparameter tuning
- Implement feature selection techniques
- Try additional machine learning models
- Enhance cross-validation strategies

## Contributing

Contributions to improve CreditSafe are welcome. Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
