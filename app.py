import streamlit as st
import pandas as pd
import joblib

# Load trained model and reference columns
model = joblib.load('best_random_forest_model.pkl')
reference_columns = joblib.load('training_columns.pkl')

st.set_page_config(page_title="Economic well being Predictor", layout="centered")
st.title("Economic well being Prediction")
st.markdown("Input region data manually or upload a CSV to predict the target outcome.")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data_encoded = pd.get_dummies(data)
    data_encoded = data_encoded.reindex(columns=reference_columns, fill_value=0)
    predictions = model.predict(data_encoded)
    data['Predicted_Target'] = predictions
    st.subheader("Batch Prediction Results")
    st.write(data)
    st.download_button("Download CSV", data.to_csv(index=False), "predictions.csv", "text/csv")

# Manual input form
with st.form("manual_input"):
    st.subheader("🔍 Manual Input")
    
    country = st.selectbox("Country", ['Nigeria', 'Kenya', 'Ghana'])
    year = st.slider("Year", min_value=2000, max_value=2025, value=2023)
    urban_rural = st.selectbox("Urban or Rural", ['Urban', 'Rural'])
    lights = st.slider("Nighttime Lights", 0.0, 100.0, 20.0)
    shoreline_dist = st.slider("Distance to Shoreline", 0.0, 100.0, 50.0)
    # ... (add other inputs here)

    submit = st.form_submit_button("🔮 Predict")

if submit:
    input_dict = {
        'country': country,
        'year': year,
        'urban_or_rural': urban_rural,
        'nighttime_lights': lights,
        'dist_to_shoreline': shoreline_dist,
        # ... (fill in the rest)
    }

    df_input = pd.DataFrame([input_dict])
    df_encoded = pd.get_dummies(df_input)
    df_encoded = df_encoded.reindex(columns=reference_columns, fill_value=0)
    prediction = model.predict(df_encoded)[0]
    st.subheader("Prediction Result")
    st.success(f"Predicted Target: {prediction}")