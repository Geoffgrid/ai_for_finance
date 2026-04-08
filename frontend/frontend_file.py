import streamlit as st
import requests

API_URL = "https://ai-for-finance-app-688958849481.europe-west1.run.app/predict"

st.title("Bitcoin Prediction (MVP_0)")

with st.form("prediction_form"):
    open_price = st.number_input("Open", placeholder="Open price")
    high = st.number_input("High", placeholder="High price")
    low = st.number_input("Low", placeholder="Low price")
    close = st.number_input("Close", placeholder="Close price")
    volume = st.number_input("Volume", min_value=0, step=1)

    submit = st.form_submit_button("Submit")

if submit:
    params = {
        "open_price": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }
    response = requests.get(API_URL, params=params)
    result = response.json()
    st.success(f"Will the price increse in the next five days : {bool(result['prediction'])}")
