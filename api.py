from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

class RequestBody(BaseModel):
    text: str

app = FastAPI()

# Load model and vectorizer
model = joblib.load("svm_3class_model.pkl")
tfidf = joblib.load("tfidf_svm_3class.pkl")

@app.post("/predict")
def predict(request: RequestBody):
    text = request.text
    # Clean incoming text (optional: use the same preprocessing as in training)
    # You can import and use your clean_text function here if needed

    # Transform with fitted vectorizer
    vector = tfidf.transform([text])

    # Predict with your SVM
    sentiment = model.predict(vector)[0]
    # Calculate confidence (modify if you want probability instead)
    confidence = float(np.max(model.decision_function(vector)))
    
    return {
        "sentiment": sentiment,
        "confidence": confidence
    }
