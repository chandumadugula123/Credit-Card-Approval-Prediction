# Credit Card Approval Prediction

This project trains machine learning classifiers to predict whether a credit card applicant should be approved or rejected. It compares Logistic Regression, Random Forest, XGBoost, and Decision Tree models, saves the best performer, and serves real-time predictions through a Flask web app.

## Features

- Applicant screening form for income, employment, credit history, loan balance, inquiries, and overdue records.
- Training pipeline with preprocessing for numeric and categorical variables.
- Binary risk-label conversion support for datasets with payment status columns.
- Model comparison report saved under `reports/`.
- IBM Watson Machine Learning deployment helper.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000` after Flask starts.

## Training With Your Dataset

Use a CSV with these columns:

`gender`, `income_type`, `education`, `family_status`, `housing_type`, `employment_status`, `annual_income`, `employment_years`, `age`, `credit_history_years`, `existing_loan_balance`, `credit_inquiries`, `past_due_count`, and `approved`.

Then run:

```bash
python train_model.py --data path\to\credit_card_applicants.csv
```

If the dataset does not have `approved` but contains payment status columns such as `STATUS`, the training script converts status codes into a binary approval label.

## IBM Watson Deployment

Set these environment variables before running the deployment helper:

```bash
set IBM_WML_API_KEY=your-api-key
set IBM_WML_URL=https://us-south.ml.cloud.ibm.com
set IBM_WML_SPACE_ID=your-space-id
python watson_deploy.py
```
