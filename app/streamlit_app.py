import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and preprocessor
model = joblib.load("C:/nmlops/tender-overshoot-mlops/scripts/models/random_forest_model.pkl")
preprocessor = joblib.load("C:/nmlops/tender-overshoot-mlops/data/processed/preprocessor.joblib")

# Title
st.title("📈 Tender Cost Overshoot Predictor")
st.markdown("Predict whether a government tender will exceed its estimated cost.")

# Sidebar for user input
st.header("📝 Tender Input Form")

description = st.text_area("Tender Description", "")
region = st.selectbox("Region", ["London", "Scotland", "North West", "South East", "Other"])
org_name = st.selectbox("Organisation Name", ["Department A", "Department B", "Council C", "Other"])

estimated_cost = st.number_input("Estimated Cost Low", min_value=0, value=100000)
final_cost = st.number_input("Estimated Cost High", min_value=0, value=120000)

# Predict button
if st.button("Predict Overshoot"):
    delta_cost = final_cost - estimated_cost

    # Construct input DataFrame
    input_df = pd.DataFrame([{
        "Description": description,
        "Region": region,
        "Organisation Name": org_name,
        "estimated_cost": estimated_cost,
        "final_cost": final_cost,
        "delta_cost": delta_cost
    }])

    # Transform features
    X_input = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    # Display result
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ This tender is likely to OVERSHOOT. (Confidence: {proba:.2f})")
    else:
        st.success(f"✅ This tender is likely to stay within budget. (Confidence: {1 - proba:.2f})")
