import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

# โหลดโมเดล Neural Network และตัวช่วยแปลงข้อมูล
model = tf.keras.models.load_model("models/ramen_model.keras")
scaler = joblib.load("models/ramen_scaler.pkl")
label_encoders = joblib.load("models/ramen_label_encoders.pkl")  # ✅ โหลด LabelEncoders

def show():
    st.title("🧪 Demo: Ramen Neural Network Model")
    st.write("🔍 เลือกข้อมูลราเมนเพื่อทำนายคะแนน (`Stars`)")

    # ✅ ให้ผู้ใช้เลือกค่าที่มีใน Dataset เท่านั้น
    brand = st.selectbox("🏭 Brand", label_encoders["Brand"].classes_)
    variety = st.selectbox("🍜 Variety", label_encoders["Variety"].classes_)
    style = st.selectbox("🥢 Style", label_encoders["Style"].classes_)
    country = st.selectbox("🌍 Country", label_encoders["Country"].classes_)

    if st.button("🔮 ทำนายคะแนนราเมน"):
        # ✅ แปลงข้อความเป็นตัวเลข
        input_data = np.array([
            label_encoders["Brand"].transform([brand])[0],
            label_encoders["Variety"].transform([variety])[0],
            label_encoders["Style"].transform([style])[0],
            label_encoders["Country"].transform([country])[0]
        ]).reshape(1, -1)

        # StandardScaler
        input_data_scaled = scaler.transform(input_data)

        # ทำนายผล
        prediction = model.predict(input_data_scaled)[0][0]
        st.success(f"⭐ โมเดลทำนายว่าราเมนนี้ได้คะแนน: **{prediction:.2f} Stars**")
