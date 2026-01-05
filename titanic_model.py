import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Titanic Prediction")

st.title("🚢 Titanic Survival Prediction")
st.write("If you can see this, Streamlit is working ✅")

# Load model
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("titanic_scaler.pkl")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
fare = st.slider("Fare", 0.0, 500.0, 32.0)

sex_val = 1 if sex == "Male" else 0

if st.button("Predict"):
    data = np.array([[pclass, sex_val, age, fare]])
    data = scaler.transform(data)
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("🎉 Passenger would SURVIVE")
    else:
        st.error("💀 Passenger would NOT survive")
