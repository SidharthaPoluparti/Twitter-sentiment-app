import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_model():
    model = joblib.load('svm_3class_model.pkl')
    tfidf = joblib.load('tfidf_svm_3class.pkl')
    return model, tfidf

model, tfidf = load_model()

st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="wide")
st.title("🎯 Twitter Sentiment Analysis (3-Class SVM)")
st.markdown("Analyze sentiment: Positive, Negative, or Neutral")
st.markdown("---")

def clean_text(text):
    if not isinstance(text, str):
        return ""
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

col1, col2 = st.columns([2, 1])
with col1:
    user_text = st.text_area("Enter tweet:", placeholder="Type text...", height=150)
with col2:
    st.info("**Steps:**\n1. Enter text\n2. Click Analyze\n3. Get result")

if st.button("🔍 Analyze", use_container_width=True):
    if user_text.strip() == "":
        st.error("Enter text")
    else:
        cleaned = clean_text(user_text)
        X_vec = tfidf.transform([cleaned])
        prediction = model.predict(X_vec)[0]
        decision_scores = model.decision_function(X_vec)[0]
        
        max_score = np.max(np.abs(decision_scores))
        confidence = (np.max(decision_scores / max_score) + 1) / 2 if max_score > 0 else 0.5
        confidence = max(0.1, min(confidence, 0.99))
        
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prediction == "Positive":
                st.success(f"✅ {prediction}")
            elif prediction == "Negative":
                st.error(f"❌ {prediction}")
            else:
                st.warning(f"⚪ {prediction}")
        with res_col2:
            st.info(f"Confidence: {confidence*100:.1f}%")
        
        st.progress(confidence)
