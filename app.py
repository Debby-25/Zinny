import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model and reference features
model = joblib.load('best_random_forest_model.pkl')
reference_features = joblib.load('training_columns.pkl')

st.set_page_config(page_title="Economic Well-Being Predictor", layout="centered")
st.title("Economic Well-Being Prediction")
st.markdown("Use the form below to enter individual feature values and get a prediction.")

# =============================
#  Step 1: Feature Input Form
# =============================
with st.form("input_form"):
    country = st.selectbox("Country", ['Cameroon', 'Ghana', 'Nigeria', 'Tanzania', 'Zimbabwe'])
    residence = st.selectbox("Residence Type", ['Urban', 'Rural'])
    age = st.slider("Age", 18, 100, 30)
    household_size = st.slider("Household Size", 1, 20, 5)
    education_level = st.selectbox("🎓 Education", ['None', 'Primary', 'Secondary', 'Tertiary'])
    employment_status = st.selectbox("Employment", ['Unemployed', 'Self-employed', 'Employed', 'Student'])
    has_bank_account = st.radio("Bank Account", ['Yes', 'No'])
    owns_property = st.radio("Owns Property", ['Yes', 'No'])
    monthly_income = st.number_input("Monthly Income", min_value=0, step=10, value=100)

    submit = st.form_submit_button("🔮 Predict")

# =============================
# Step 2: Preprocess + Predict
# =============================
if submit:
    input_dict = {
        'country': country,
        'residence_type': residence,
        'age': age,
        'household_size': household_size,
        'education_level': education_level,
        'employment_status': employment_status,
        'has_bank_account': 1 if has_bank_account == 'Yes' else 0,
        'owns_property': 1 if owns_property == 'Yes' else 0,
        'monthly_income': monthly_income
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=reference_features, fill_value=0)
    st.write(input_df)
    prediction = model.predict(input_df)[0]

    # =============================
    # ✅ Step 3: Display Result
    # =============================
    st.subheader("Result")
    if prediction == 1:
        st.success("The predicted well-being status is: **Likely Above Threshold**")
    else:
        st.warning("The predicted well-being status is: **Likely Below Threshold**")