import streamlit as st
import requests

st.title("🍎 Fruit Freshness Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
API_URL = "https://your-fastapi-service.onrender.com/predict"
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image")

    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        
        response = requests.post(
            API_URL,
            files={"file": uploaded_file.getvalue()}
        )

        result = response.json()

        st.success(f"Prediction: {result['class']}")
        st.write(f"Confidence: {result['confidence']*100:.2f}%")