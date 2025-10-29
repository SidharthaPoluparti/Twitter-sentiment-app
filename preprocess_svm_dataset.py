import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load raw dataset WITHOUT headers and assign column names
df = pd.read_csv('twitter_sentiment.csv', header=None, names=['tweet_id', 'entity', 'sentiment', 'text'])

print("Original dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nOriginal class distribution:")
print(df['sentiment'].value_counts())

# Keep only Positive, Negative, Neutral (remove Irrelevant)
df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

print("\nAfter filtering (removed Irrelevant):")
print(df['sentiment'].value_counts())

# Text cleaning function
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

# Apply cleaning to text column
df['clean_text'] = df['text'].apply(clean_text)

# Remove rows with empty cleaned text
df = df[df['clean_text'].str.strip() != '']

print("\nAfter text cleaning:")
print("Final dataset shape:", df.shape)
print("Final class distribution:")
print(df['sentiment'].value_counts())

# Save final dataset
df.to_csv('final_twitter_sentiment_svm.csv', index=False)

print("\n✅ Dataset saved as 'final_twitter_sentiment_svm.csv'")
