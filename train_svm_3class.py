import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the 3-class SVM dataset
df = pd.read_csv('final_twitter_sentiment_svm.csv')

print("Dataset loaded:")
print(f"Total samples: {len(df)}")
print("\nClass distribution:")
print(df['sentiment'].value_counts())
print()

# Prepare features and labels
X = df['clean_text']
y = df['sentiment']

# Vectorize text using TF-IDF
print("Vectorizing text with TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_vec = tfidf.fit_transform(X)

# Split into train and test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print()

# Train SVM model (3-class: Positive, Negative, Neutral)
print("Training Linear SVM on 3-class problem...")
model = LinearSVC(max_iter=2000, random_state=42, dual=True)
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, 'svm_3class_model.pkl')
joblib.dump(tfidf, 'tfidf_svm_3class.pkl')
print("✅ Model and vectorizer saved")
print()

# Evaluate on training data
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Evaluate on test data
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print results
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print()
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred))
