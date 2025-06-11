import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')  # Bisa ColumnTransformer atau Pipeline

st.title("Prediksi Heart Failure")

with st.form("form_hf"):
    age        = st.number_input('Age', min_value=40, max_value=95, value=60)
    anaemia    = st.selectbox('Anemia?', options=[0,1])
    high_bp    = st.selectbox('High Blood Pressure?', options=[0,1])
    creatinine = st.number_input('Serum Creatinine', min_value=0.5, max_value=3.0, step=0.01, value=1.0)
    ejection   = st.number_input('Ejection Fraction', min_value=10, max_value=80, value=30)
    platelets  = st.number_input('Platelets (k/µL)', min_value=100000, max_value=500000, value=250000)
    submit     = st.form_submit_button("Prediksi")

if submit:
    # Susun input ke array
    X = np.array([[age, anaemia, high_bp, creatinine, ejection, platelets]])

    # Transformasi input pakai scaler (ColumnTransformer)
    X_transformed = scaler.transform(X)

    # Prediksi
    prediction = model.predict(X_transformed)[0]

    # Output
    if prediction == 1:
        st.error("Hasil: Risiko Heart Failure tinggi")
    else:
        st.success("Hasil: Risiko Heart Failure rendah")
