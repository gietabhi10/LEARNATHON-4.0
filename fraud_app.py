import streamlit as st
import joblib
import pandas as pd

model = joblib.load("fraud_detection_model.pkl")

st.title("🚨 Auto Insurance Fraud Detection")

vehicle_age = st.number_input("Vehicle Age")
policy_premium = st.number_input("Policy Premium")
total_claim = st.number_input("Total Claim Amount")

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Vehicle_Age": vehicle_age,
        "Policy_Premium": policy_premium,
        "Total_Claim": total_claim
    }])
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.write(f"Prediction: **{'🚨 Fraud' if prediction else '✅ Not Fraud'}**")
    st.write(f"Fraud Probability: {prob:.2f}")
