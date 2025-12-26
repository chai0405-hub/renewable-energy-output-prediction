import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model/trained_model.pkl", "rb"))

st.title("Renewable Energy Output Prediction")

temp = st.number_input("Temperature (°C)")
wind = st.number_input("Wind Speed (m/s)")
solar = st.number_input("Solar Irradiance (W/m²)")

if st.button("Predict Energy Output"):
    result = model.predict([[temp, wind, solar]])
    st.success(f"Predicted Energy Output: {result[0]:.2f} kWh")
