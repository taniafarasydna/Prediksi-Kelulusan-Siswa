import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ===============================
# LOAD MODEL & SCALER
# ===============================
model = load_model("model_kelulusan.h5")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Kelulusan Siswa Matematika")
st.write("Aplikasi ini menggunakan model Artificial Neural Network (ANN) untuk memprediksi kelulusan berdasarkan data siswa.")


# ===============================
# INPUT USER
# ===============================

# Semua fitur penting dari dataset student-mat
school = st.selectbox("School", ["GP", "MS"])
sex = st.selectbox("Sex", ["F", "M"])
age = st.slider("Age", 15, 22, 17)
address = st.selectbox("Address", ["U", "R"])
famsize = st.selectbox("Family Size", ["LE3", "GT3"])
Pstatus = st.selectbox("Parent Status", ["T", "A"])
studytime = st.slider("Study Time (1–4)", 1, 4, 2)
failures = st.slider("Past Failures", 0, 3, 0)
famrel = st.slider("Family Relationship (1–5)", 1, 5, 4)
freetime = st.slider("Free Time (1–5)", 1, 5, 3)
goout = st.slider("Going Out (1–5)", 1, 5, 3)
Dalc = st.slider("Daily Alcohol", 1, 5, 1)
Walc = st.slider("Weekend Alcohol", 1, 5, 1)
health = st.slider("Health (1–5)", 1, 5, 3)
absences = st.slider("Absences", 0, 93, 2)


# ===============================
# KONVERSI KE DATAFRAME
# ===============================
input_data = {
    "school": school,
    "sex": sex,
    "age": age,
    "address": address,
    "famsize": famsize,
    "Pstatus": Pstatus,
    "studytime": studytime,
    "failures": failures,
    "famrel": famrel,
    "freetime": freetime,
    "goout": goout,
    "Dalc": Dalc,
    "Walc": Walc,
    "health": health,
    "absences": absences
}

df_input = pd.DataFrame([input_data])

# One-hot encoding sesuai training
df_input = pd.get_dummies(df_input)
df_input = df_input.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Scaling
X_scaled = scaler.transform(df_input)

# ===============================
# PREDIKSI
# ===============================
if st.button("Prediksi Kelulusan"):
    pred = model.predict(X_scaled)[0][0]

    if pred > 0.5:
        st.success(f"Hasil: LULUS (Probabilitas: {pred:.2f})")
    else:
        st.error(f"Hasil: TIDAK LULUS (Probabilitas: {pred:.2f})")
