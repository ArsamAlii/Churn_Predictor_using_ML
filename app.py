import streamlit as st
import joblib
from ui import get_user_input


st.title("Telecom Customer Churn Prediction")


# Fix monotonic_cst sklearn issue
def fix_monotonic_issue(model):
    if not hasattr(model, "monotonic_cst"):
        try:
            model.monotonic_cst = None
        except:
            pass

    if hasattr(model, "estimators_"):
        for estimator in model.estimators_:
            fix_monotonic_issue(estimator)

    if hasattr(model, "named_steps"):
        for name, step in model.named_steps.items():
            fix_monotonic_issue(step)

    return model


# Simple explanation without API
def simple_explanation(row, prediction, prob):
    reasons = []

    if row["Contract"] == "Month-to-month":
        reasons.append("The customer has a month-to-month contract, so they can leave easily. This increases churn risk.")
    else:
        reasons.append("The customer has a longer contract, which usually reduces churn risk.")

    if row["tenure"] <= 12:
        reasons.append("The customer has low tenure, meaning they are still new and may not be loyal yet.")
    elif row["tenure"] >= 36:
        reasons.append("The customer has high tenure, which usually shows stronger loyalty.")

    if row["MonthlyCharges"] >= 80:
        reasons.append("The monthly charges are high, which can increase dissatisfaction.")
    else:
        reasons.append("The monthly charges are not very high, so price pressure is lower.")

    if row["InternetService"] == "Fiber optic":
        reasons.append("Fiber optic service can increase churn risk if the customer feels the cost is high.")
    elif row["InternetService"] == "DSL":
        reasons.append("DSL customers may have moderate churn risk depending on speed and service quality.")
    elif row["InternetService"] == "No":
        reasons.append("The customer has no internet service, so internet-related complaints are less likely.")

    if row["PaymentMethod"] == "Electronic check":
        reasons.append("Electronic check payment is often linked with higher churn risk.")
    elif "automatic" in row["PaymentMethod"]:
        reasons.append("Automatic payment usually shows convenience and lower churn risk.")

    if row["SeniorCitizen"] == 1:
        reasons.append("The customer is a senior citizen, so price sensitivity and support quality may affect churn risk.")

    result = "likely to churn" if prediction == 1 else "not likely to churn"

    return f"The model predicted that this customer is {result} with probability {prob:.2f}. " + " ".join(reasons)


# Load model
pipeline = joblib.load("model_pipeline.joblib")
pipeline = fix_monotonic_issue(pipeline)


# Get user input
input_data = get_user_input()


# Predict button
if st.button("Predict Churn"):
    try:
        prediction = pipeline.predict(input_data)[0]
        prob = pipeline.predict_proba(input_data)[0][1]

        row_dict = input_data.iloc[0].to_dict()

        if prediction == 1:
            st.error(f"Customer Likely to Churn! Probability: {prob:.2f}")
        else:
            st.success(f"Customer Not Likely to Churn. Probability: {prob:.2f}")

        explanation = simple_explanation(row_dict, prediction, prob)
        st.info(f"Model Explanation: {explanation}")

    except Exception as e:
        st.warning(f"Model Error: {e}")