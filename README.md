# Telecom Customer Churn Prediction App

This project is a Streamlit-based machine learning app for predicting telecom customer churn. It takes customer information as input, uses a trained model pipeline to predict whether the customer is likely to churn, and also provides a simple AI-generated explanation for the prediction.

## Features

- Predicts whether a telecom customer is likely to churn
- Uses customer details such as tenure, monthly charges, internet service, contract type, payment method, and senior citizen status
- Displays churn probability
- Provides an explanation of the prediction
- Simple and interactive Streamlit user interface

## Project Files

- `app.py` - main Streamlit application
- `ui.py` - handles user input form
- `model_pipeline.joblib` - saved trained machine learning pipeline
- `main.ipynb` - notebook used for model training or experimentation
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` - dataset used for the project

## Technologies Used

- Python
- Streamlit
- Pandas
- Joblib
- Scikit-learn
- LangChain
- Google Generative AI

## How It Works

1. The user enters customer information in the Streamlit app.
2. The saved machine learning pipeline loads the input data.
3. The model predicts whether the customer is likely to churn or not.
4. The app shows the churn probability.
5. An AI-generated explanation is displayed to describe why the model may have made that prediction.

## Input Features

The model uses the following customer details:

- `tenure`
- `MonthlyCharges`
- `InternetService`
- `Contract`
- `PaymentMethod`
- `SeniorCitizen`

## Requirements

Install the required libraries before running the project:

```bash
pip install streamlit pandas joblib scikit-learn langchain langchain-google-genai
