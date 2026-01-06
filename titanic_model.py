import streamlit as st
import numpy as np
import joblib  # to load model
import pandas as pd


model = joblib.load("titanic_model.pkl")  # your model file

st.set_page_config(page_title="Titanic Survival Predictor 🚢", layout="centered")
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict whether they would survive or not:")


col1, col2 = st.columns(2)

with col1:
    Pclass = st.selectbox("Passenger Class", [1, 2, 3])
    Sex = st.selectbox("Sex", ["male", "female"])
    Age = st.number_input("Age", min_value=0, max_value=100, value=25)

with col2:
    SibSp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
    Parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=0)
    Fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=32.0)

Sex_numeric = 0 if Sex == "male" else 1



input_data = np.array([[Fare]])  # only the feature your model was trained on


if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None

        if prediction[0] == 1:
            st.success("✅ The passenger is likely to survive!")
            if probability is not None:
                st.info(f"Survival probability: {probability[0][1]*100:.2f}%")
        else:
            st.error("❌ The passenger is not likely to survive!")
            if probability is not None:
                st.info(f"Survival probability: {probability[0][0]*100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


if st.checkbox("Show sample dataset"):
    df = pd.read_csv("titanic.csv")
    st.dataframe(df.head(10))

