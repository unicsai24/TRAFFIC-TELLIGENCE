import streamlit as st
import pandas as pd
import joblib

st.title('Traffic Volume Predictor')

uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    model = joblib.load('model.pkl')
    predictions = model.predict(input_df)
    st.write('Predicted Traffic Volume:', predictions)
