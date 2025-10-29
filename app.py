import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load trained 3-class model and vectorizer
model = joblib.load('svm_3class_model.pkl')
tfidf = joblib.load('tfidf_svm_3class.pkl')

# Initialize FastAPI app
app = FastAPI(title="Twitter Sentiment Analysis API (3-Class)")

# Define request model
class SentimentRequest(BaseModel):
    text: str

# Define response model
class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

# Define text cleaning function
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

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Twitter Sentiment Analysis API (3-Class) is running"}

# Prediction endpoint
@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    # Clean the input text
    cleaned_text = clean_text(request.text)
    
    # Vectorize the cleaned text
    text_vectorized = tfidf.transform([cleaned_text])
    
    # Get prediction
    prediction = model.predict(text_vectorized)[0]
    
    # Get confidence score using decision function
    decision_scores = model.decision_function(text_vectorized)[0]
    
    # Normalize decision scores to confidence (0-1 range)
    max_score = np.max(np.abs(decision_scores))
    if max_score > 0:
        normalized_scores = decision_scores / max_score
        confidence = (np.max(normalized_scores) + 1) / 2  # Scale to 0-1
    else:
        confidence = 0.5
    
    confidence = max(0.1, min(confidence, 0.99))  # Keep between 0.1 and 0.99
    
    return SentimentResponse(
        text=request.text,
        sentiment=prediction,
        confidence=round(confidence, 2)
    )
