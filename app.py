import streamlit as st
import joblib
from ui import get_user_input
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key="AIzaSyBMstdrPyABv_vMviTd5gDlBp4fV6vhWT0")

system_msg = SystemMessage(content="""
You are an expert AI model explainer. Your task is to clearly and logically explain why a randomforest model made a specific prediction.
There are 5 column in the dataset: tenure (ranging from 1-72), MonthlyCharges (ranging from 18.25-118.75), InternetService (categorical: 'DSL', 'Fiber optic', 'No'), 
Contract (categorical: 'Month-to-month', 'One year', 'Two year'), 
PaymentMethod (categorical: 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'), SeniorCitizen (binary: 0, 1).
Tell me why the model predicted this result and on which criteria the model made this decision.
Give reason according to company policies like PTCL,STORMFIBER and their standards and based on stats and customer satisfaction.
Use simple, professional English and keep the explanation concise and factual.
""")

st.title("Telecom Customer Churn Prediction")
input_data = get_user_input()
pipeline = joblib.load("model_pipeline.joblib")

if st.button("Predict Churn"):
    try:
        prediction = pipeline.predict(input_data)[0]
        prob = pipeline.predict_proba(input_data)[0][1]

        row_dict = input_data.iloc[0].to_dict()

        human_msg = HumanMessage(content=f"""
        The random forest predicted: {prediction}
        Input data:
        {row_dict}
        Explain why the model predicted this result.
        """)

        if prediction == 1:
            st.error(f"Customer Likely to Churn! Probability: {prob:.2f}")
        else:
            st.success(f"Customer Not Likely to Churn. Probability: {prob:.2f}")

        response = llm.invoke([system_msg,human_msg])
        st.info(f"Model Explanation: {response.content}")

    except Exception as e:
        st.warning(f"Model Error: {e}")
