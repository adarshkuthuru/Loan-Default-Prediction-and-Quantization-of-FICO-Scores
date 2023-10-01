# Loan Default Prediction and Quantization of FICO Scores

This repository contains predictive models that estimate the probability of a borrower defaulting on a loan based on several features of the borrower, as well as a quantization technique for bucketing FICO scores for better predictive performance with categorical inputs.

## Data

The dataset used is `Task 3 and 4_Loan_Data.csv` which has the following columns:

- `customer_id`: Unique ID of the customer.
- `credit_lines_outstanding`: Number of active credit lines.
- `loan_amt_outstanding`: Outstanding loan amount.
- `total_debt_outstanding`: Total debt amount.
- `income`: Borrower's income.
- `years_employed`: Number of years employed.
- `fico_score`: Borrower's FICO score.
- `default`: Default status (1 if the borrower defaulted, 0 otherwise).

## Usage

### 1. Probability of Default and Expected Loss Calculation:

File: `Prob_Default_Expected_Loss.py`

#### Load and Preprocess Data

```python
from Prob_Default_Expected_Loss import load_loan_data, preprocess_loan_data
data = load_loan_data("<path_to_csv_file>")
X_train, X_test, y_train, y_test, scaler = preprocess_loan_data(data)
```

#### Train Models and Evaluate

```python
from Prob_Default_Expected_Loss import train_and_evaluate_models
models, auc_scores = train_and_evaluate_models(X_train, X_test, y_train, y_test)
```

#### Compute Expected Loss for a Sample Data

```python
from Prob_Default_Expected_Loss import compute_expected_loss

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
```

### 2. Quantization of FICO Scores:

File: `Quantization.py`

#### Load and Preprocess Data (Quantized FICO Scores)

```python
from Quantization import load_loan_data, preprocess_loan_data
data = load_loan_data("<path_to_csv_file>")
X_train, X_test, y_train, y_test, scaler = preprocess_loan_data(data, n_buckets=10)
```

#### Train Models and Evaluate (using Quantized Data)

```python
from Quantization import train_and_evaluate_models
models, auc_scores = train_and_evaluate_models(X_train, X_test, y_train, y_test)
```

#### Compute Expected Loss for a Sample Data (using Quantized Data)

```python
from Quantization import compute_expected_loss

sample_loan_data = pd.DataFrame({
    'credit_lines_outstanding': [5],
    'loan_amt_outstanding': [15000],
    'total_debt_outstanding': [25000],
    'income': [55000],
    'years_employed': [5],
    'fico_score': [700]  # This will be quantized automatically
})

loss = compute_expected_loss(models['Random Forest'], sample_loan_data, scaler)
print(f"Expected Loss: ${loss:,.2f}")
```

## Context

Banks rely on accurate predictions to set aside sufficient capital for potential loan losses. Predictive models in this repository assist in evaluating the default risk associated with borrowers based on various metrics like income, outstanding debt, and FICO scores. For mortgages, as FICO scores are continuous and may not directly fit into certain models, quantization is used to bucket them into meaningful categories to help with predictions.

## Contribution

Pull requests are welcome! Ensure that changes passed all tests before submitting a PR.
