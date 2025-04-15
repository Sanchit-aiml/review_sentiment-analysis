# app/app.py - Streamlit frontend

import streamlit as st
import pickle
import re

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    return text

import os

model_path = os.path.join("models", "review sentiment_classifier_model.pkl")
model = pickle.load(open(model_path, "rb"))

import os

vectorizer_path = os.path.join("models", "review sentiment_classifier_vectorizer.pkl")
vectorizer = pickle.load(open(vectorizer_path, "rb"))


st.title("Amazon Review Sentiment Analyzer")
user_input = st.text_area("Enter a product review")

if st.button("Analyze"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        st.write(f"**Predicted Sentiment: {prediction}**")
    else:
        st.write("Please enter a valid review.")
