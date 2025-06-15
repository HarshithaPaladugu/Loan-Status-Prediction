import streamlit as st
import numpy as np
import pickle

# Load the saved model
with open('loan_status_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title
st.title("Loan Status Prediction")

# User inputs
no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
income_annum = st.number_input("Annual Income (in ₹)", min_value=0)
loan_amount = st.number_input("Loan Amount (in ₹)", min_value=0)
loan_term = st.number_input("Loan Term (in Years)", min_value=0)
cibil_score = st.number_input("CIBIL Score", min_value=0, max_value=900)
residential_assets_value = st.number_input("Residential Asset Value (in ₹)", min_value=0)
commercial_assets_value = st.number_input("Commercial Asset Value (in ₹)", min_value=0)
luxury_assets_value = st.number_input("Luxury Asset Value (in ₹)", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value (in ₹)", min_value=0)
education = st.selectbox("Education", options=[0, 1], format_func=lambda x: 'Graduate' if x == 1 else 'Not Graduate')
self_employed = st.selectbox("Self Employed", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Predict button
if st.button("Predict Loan Status"):
    features = np.array([[no_of_dependents, income_annum, loan_amount, loan_term,
                          cibil_score, residential_assets_value, commercial_assets_value,
                          luxury_assets_value, bank_asset_value, education, self_employed]])
    
    prediction = model.predict(features)

    result = "Approved" if prediction[0] == 1 else "Rejected"
    st.success(f"Loan Status: {result}")
