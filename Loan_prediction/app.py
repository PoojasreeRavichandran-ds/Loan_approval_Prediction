import pandas as pd
df = pd.read_csv(r"C:\Users\ELCOT\Desktop\Loan_prediction\loan_approval_dataset.csv")
print(df.head())

df.columns = df.columns.str.strip()

import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("loan_approval_model.pkl")

st.title("🏦 Loan Approval Prediction")
st.write("Model Loaded Successfully!")

income_annum = st.number_input(
    "Annual Income",
    min_value=0,
    value=500000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=1000000
)

loan_term = st.number_input(
    "Loan Term (Years)",
    min_value=1,
    value=10
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=700
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["No", "Yes"]
)

# Same encoding used in main.py
education_encoded = 1 if education == "Graduate" else 0
self_employed_encoded = 1 if self_employed == "Yes" else 0

if st.button("Predict Loan Status"):

    input_data = np.array([[
        self_employed_encoded,
        education_encoded,
        loan_amount,
        loan_term,
        cibil_score,
        income_annum
    ]])

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    st.write("Approval Probability:", round(probability[0][1] * 100, 2), "%")

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")