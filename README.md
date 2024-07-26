# CreditSafe: A Python-Driven Credit Risk Assessment Model

## Project Overview

CreditSafe is a machine learning project designed to assess credit risk using Python. It demonstrates skills in data preprocessing, SQL database interaction, and the implementation of machine learning models for classification tasks.

## Features

- Data preparation and cleaning
- SQL database integration using SQLite
- Implementation of Logistic Regression and Random Forest classifiers
- Model evaluation using various metrics (accuracy, precision, recall, F1-score, AUC-ROC)
- Cross-validation for robust model assessment

## Project Structure
```
creditsafe/
│
├── main.py
├── data_preparation.py
├── model_development.py
├── view_database.py
├── requirements.txt
├── credit_risk.db
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Steps

1. Clone the repository:

```git clone git@gitlab.com:jmc-gl/creditsafe.git && cd creditsafe```

2. Create a virtual environment:

   - On macOS and Linux:
     ```
     python3 -m venv creditsafe_env
     source creditsafe_env/bin/activate
     ```
   - On Windows:
     ```
     python -m venv creditsafe_env
     creditsafe_env\Scripts\activate
     ```

3. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

1. Prepare the data:
```
python data_preparation.py
```
2. Run the main script:
```
python main.py
```

3. Run the view database script:
```
python view_database.py
```

This will execute the entire pipeline: data preparation, model training, evaluation, and storing results in the SQLite database.

## Results

The model results are stored in the `model_results` table of the `credit_risk.db` SQLite database. You can query this table to view the performance metrics of both the Logistic Regression and Random Forest models.

## Troubleshooting

- If you encounter a "command not found" error, ensure that Python is installed and added to your system's PATH.
- On macOS or Linux, you might need to use `python3` instead of `python` if both Python 2 and 3 are installed.
- If you're using an IDE, make sure it's configured to use the virtual environment you created.

## Contributing

Contributions to improve CreditSafe are welcome. Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
