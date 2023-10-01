#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import Dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def load_loan_data(file_path):
    """ Load the loan data from a CSV file and handle missing values. """
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)
    return data

def quantize_fico(fico_scores, n_buckets):
    """ Quantize FICO scores into a specified number of buckets. """
    labels = range(1, n_buckets + 1)
    fico_quantized = pd.qcut(fico_scores, n_buckets, labels=labels)
    fico_quantized = n_buckets + 1 - fico_quantized.astype(int)
    return fico_quantized

def preprocess_loan_data(data, n_buckets=5):
    """ Split data into features and target, scale, and quantize FICO scores. """
    X = data.drop(columns=['customer_id', 'default'])
    y = data['default']

    X['fico_score'] = quantize_fico(X['fico_score'], n_buckets)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """ Train various models and evaluate their performance. """
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
    """ Compute expected loss for a given loan data. """
    data_scaled = scaler.transform(data)
    prob_default = model.predict_proba(data_scaled)[:, 1]
    
    loan_amt = data['loan_amt_outstanding'].values
    expected_loss = loan_amt * prob_default * (1 - recovery_rate)
    return expected_loss[0]

if __name__ == '__main__':
    data_path = r"C:\Users\adars\Downloads\Laptop\Drive\Programming_resources\Python\JPM Online Internship\Task-3\Task 3 and 4_Loan_Data.csv"
    data = load_loan_data(data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_loan_data(data, n_buckets=10)

    models, auc_scores = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    sample_loan_data = pd.DataFrame({
        'credit_lines_outstanding': [5],
        'loan_amt_outstanding': [15000],
        'total_debt_outstanding': [25000],
        'income': [55000],
        'years_employed': [5],
        'fico_score': [700]
    })
    
    loss = compute_expected_loss(models['Random Forest'], sample_loan_data, scaler)
    print(f"Expected Loss: ${loss:,.2f}")
