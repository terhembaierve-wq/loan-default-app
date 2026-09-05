"""
STAGE 6: APPLICATION & INTEGRATION
Goal: Deploy the recommended model as a Streamlit prototype.
Run with: streamlit run 06_streamlit_app.py
"""

import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved preprocessing pipeline and final selected model
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("final_model.pkl")

st.title("Loan Default Risk Predictor")
st.write("Enter applicant details to estimate the probability of loan default.")

# 2. Collect applicant inputs through simple form widgets
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 35)
    income = st.number_input("Annual Income ($)", 0, 500000, 50000)
    loan_amount = st.number_input("Loan Amount ($)", 0, 500000, 20000)
    credit_score = st.number_input("Credit Score", 300, 850, 650)
    months_employed = st.number_input("Months Employed", 0, 600, 24)
    num_credit_lines = st.number_input("Number of Credit Lines", 0, 20, 3)
with col2:
    interest_rate = st.number_input("Interest Rate (%)", 0.0, 50.0, 10.0)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    dti_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

has_mortgage = st.radio("Has Mortgage?", ["Yes", "No"], horizontal=True)
has_dependents = st.radio("Has Dependents?", ["Yes", "No"], horizontal=True)
loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
has_cosigner = st.radio("Has Co-Signer?", ["Yes", "No"], horizontal=True)

# 3. Assemble inputs into a single-row DataFrame matching the training schema
input_df = pd.DataFrame([{
    "Age": age, "Income": income, "LoanAmount": loan_amount, "CreditScore": credit_score,
    "MonthsEmployed": months_employed, "NumCreditLines": num_credit_lines,
    "InterestRate": interest_rate, "LoanTerm": loan_term, "DTIRatio": dti_ratio,
    "Education": education, "EmploymentType": employment_type, "MaritalStatus": marital_status,
    "HasMortgage": has_mortgage, "HasDependents": has_dependents,
    "LoanPurpose": loan_purpose, "HasCoSigner": has_cosigner
}])

# 4. Predict on button click
if st.button("Predict Default Risk"):
    X_input = preprocessor.transform(input_df)         # apply the same pipeline as training
    prob_default = model.predict_proba(X_input)[0, 1]  # probability of class 1 (Default)

    st.metric("Predicted Default Probability", f"{prob_default:.1%}")

    # 5. Simple risk banding for interpretability
    if prob_default >= 0.5:
        st.error("High Risk: this applicant is likely to default.")
    elif prob_default >= 0.3:
        st.warning("Medium Risk: review recommended.")
    else:
        st.success("Low Risk: applicant is unlikely to default.")
