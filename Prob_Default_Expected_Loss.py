#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import Dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def load_loan_data(file_path):
    """
    Load the loan data from a CSV file and handle missing values.

    Parameters:
    :param file_path: str - Full path to the CSV file

    Returns:
    :return: data: pd.DataFrame - Loaded and cleaned data
    """
    # Load data from CSV file
    data = pd.read_csv(file_path)
    
    # Handle missing values (impute or remove)
    data.dropna(inplace=True)
    return data

def preprocess_loan_data(data):
    """
    Split the data into features and target, scale the features.

    Parameters:
    :param data: pd.DataFrame - Loan data

    Returns:
    :return: X_train: pd.DataFrame - Scaled training data
    :return: X_test: pd.DataFrame - Scaled testing data
    :return: y_train: pd.Series - Target for training data
    :return: y_test: pd.Series - Target for testing data
    :return: scaler: StandardScaler - The scaler used for preprocessing
    """
    # Split data into features and target
    X = data.drop(columns=['customer_id', 'default'])
    y = data['default']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train various models and evaluate their performance.

    Parameters:
    :param X_train, X_test, y_train, y_test: Data and target for training and testing.

    Returns:
    :return: dict - A dictionary of models and their respective AUC scores.
    """
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    
    auc_scores = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        auc_scores[model_name] = auc
        
        print(f"{model_name} AUC: {auc:.4f}")
    
    return models, auc_scores

def compute_expected_loss(model, data, scaler, recovery_rate=0.1):
    """
    Compute the expected loss for a given loan data.

    Parameters:
    :param model: The trained predictive model
    :param data: pd.DataFrame - The input data for prediction
    :param scaler: StandardScaler - The scaler used for preprocessing
    :param recovery_rate: float - The rate of recovery on default

    Returns:
    :return: float - Expected loss for the given data
    """
    # Ensure data is scaled
    data_scaled = scaler.transform(data)
    
    # Predict the probability of default
    prob_default = model.predict_proba(data_scaled)[:, 1]
    
    # Compute expected loss
    loan_amt = data['loan_amt_outstanding'].values
    expected_loss = loan_amt * prob_default * (1 - recovery_rate)
    
    return expected_loss[0]

if __name__ == '__main__':
    # Input full path of CSV file
    data_path = r"C:\Users\adars\Downloads\Laptop\Drive\Programming_resources\Python\JPM Online Internship\Task-3\Task 3 and 4_Loan_Data.csv"
    
    # Load and preprocess data
    data = load_loan_data(data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_loan_data(data)

    # Train models and get their AUC scores
    models, auc_scores = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Compute expected loss for a sample loan data
    sample_loan_data = pd.DataFrame({
        'credit_lines_outstanding': [5],
        'loan_amt_outstanding': [15000],
        'total_debt_outstanding': [25000],
        'income': [55000],
        'years_employed': [5],
        'fico_score': [700]
    })
    
    # Using Random Forest as the predictive model in this example
    loss = compute_expected_loss(models['Random Forest'], sample_loan_data, scaler)
    print(f"Expected Loss: ${loss:,.2f}")
