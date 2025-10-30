import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_model():
    model = joblib.load("svm_3class_model.pkl")
    tfidf = joblib.load("tfidf_svm_3class.pkl")
    return model, tfidf

model, tfidf = load_model()

def clean_text(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(filtered_words)

st.title("🎯 Twitter Sentiment Analysis App")

user_text = st.text_area("Enter tweet text:")

if st.button("🔍 Analyze Sentiment"):
    cleaned = clean_text(user_text)
    X_vec = tfidf.transform([cleaned])
    prediction = model.predict(X_vec)[0]
    st.write(f"Sentiment: {prediction}")
