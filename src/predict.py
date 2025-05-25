import streamlit as st
import pandas as pd
import joblib

# Load trained model and columns
model = joblib.load(r"D:\\Codes\\Projects\\ML\\Limi\\models\\random_forest_model.pkl")
columns = joblib.load(r"D:\\Codes\\Projects\\ML\\Limi\\models\\model_columns.pkl")

# Set page config
st.set_page_config(page_title="Road Accident Severity Predictor", layout="centered")

# App title
st.title("🚧 Road Accident Severity Predictor")

st.markdown("Enter the accident details in the sidebar to predict severity.")

# Sidebar inputs
st.sidebar.header("Accident Input Details")

vehicles = st.sidebar.number_input("Number of vehicles involved", min_value=1, max_value=10, step=1)
casualties = st.sidebar.number_input("Number of casualties", min_value=0, max_value=20, step=1)
hour = st.sidebar.slider("Hour of accident (0-23)", 0, 23, 12)
day_of_week = st.sidebar.selectbox("Day of week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
cause = st.sidebar.selectbox("Cause of accident", [
    'Overspeed', 'Overtaking', 'Changing lane',
    'No priority to pedestrian', 'Others'
])  # Replace with your actual options if different

# Build input dict with all 0s
input_data = {col: 0 for col in columns}

# Fill values
input_data['Number_of_vehicles_involved'] = vehicles
input_data['Number_of_casualties'] = casualties
input_data['Hour'] = hour

# Handle one-hot encoded categorical inputs
day_col = f'Day_of_week_{day_of_week}'
cause_col = f'Cause_of_accident_{cause}'

if day_col in input_data:
    input_data[day_col] = 1
else:
    st.warning(f"Day column '{day_col}' not found in trained model columns.")

if cause_col in input_data:
    input_data[cause_col] = 1
else:
    st.warning(f"Cause column '{cause_col}' not found in trained model columns.")

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction
if st.button("🔍 Predict Severity"):
    try:
        prediction = model.predict(input_df)[0]
        severity_labels = ['Slight Injury', 'Serious Injury', 'Fatal Injury']
        st.success(f"🚑 Predicted Severity: **{severity_labels[prediction]}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
