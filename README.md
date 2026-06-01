# Telecom Customer Churn Prediction

This project is a Machine Learning web application built with Streamlit. It predicts whether a telecom customer is likely to churn based on customer details such as tenure, monthly charges, internet service type, contract type, payment method, and senior citizen status.

The application uses a trained machine learning pipeline saved as `model_pipeline.joblib`. After making a prediction, the app also provides a simple explanation of why the customer may or may not churn.

## Features

* Predicts telecom customer churn using a trained machine learning model
* Simple Streamlit web interface
* Takes customer information as input
* Displays churn probability
* Shows whether the customer is likely to churn or not
* Provides a basic explanation of the prediction
* Includes a compatibility fix for the `monotonic_cst` scikit-learn issue

## Project Files

```text
.
├── app.py                  # Main Streamlit application
├── ui.py                   # User input interface
├── main.ipynb              # Model training / development notebook
├── model_pipeline.joblib   # Saved trained ML pipeline
└── README.md               # Project documentation
```

## Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-learn
* Joblib

## Input Features

The app uses the following customer features:

| Feature         | Description                                               |
| --------------- | --------------------------------------------------------- |
| tenure          | Number of months the customer has stayed with the company |
| MonthlyCharges  | Monthly bill amount                                       |
| InternetService | Type of internet service used by the customer             |
| Contract        | Customer contract type                                    |
| PaymentMethod   | Customer payment method                                   |
| SeniorCitizen   | Whether the customer is a senior citizen or not           |

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/telecom-customer-churn-prediction.git
cd telecom-customer-churn-prediction
```

### 2. Install Required Libraries

```bash
pip install streamlit pandas scikit-learn joblib
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

## How the App Works

1. The user enters customer details in the Streamlit interface.
2. The saved machine learning pipeline loads from `model_pipeline.joblib`.
3. The model predicts whether the customer is likely to churn.
4. The app displays the churn prediction and churn probability.
5. A simple explanation is generated based on the customer’s input values.

## Example Output

```text
Customer Likely to Churn! Probability: 0.78
```

or

```text
Customer Not Likely to Churn. Probability: 0.24
```

The app also gives an explanation, such as:

```text
The customer has a month-to-month contract, so they can leave easily. This increases churn risk.
The monthly charges are high, which can increase dissatisfaction.
```

## Model Explanation Logic

The explanation is based on common churn-related factors:

* Month-to-month contracts usually increase churn risk.
* Higher monthly charges can increase customer dissatisfaction.
* Low tenure customers are usually less loyal.
* Automatic payment methods may reduce churn risk.
* Senior citizens may be more sensitive to price and service quality.
* Internet service type can influence customer satisfaction.

## Note

The file `model_pipeline.joblib` is required to run the application. Make sure it is present in the same folder as `app.py`.

If you face a scikit-learn compatibility error related to `monotonic_cst`, the app already includes a small fix to handle this issue.

## Future Improvements

* Add more detailed model evaluation metrics
* Add visualizations for feature importance
* Improve explanation using SHAP or LIME
* Add model comparison between Logistic Regression, Decision Tree, and Random Forest
* Deploy the app online using Streamlit Community Cloud

## Author

Developed by Arsam.
