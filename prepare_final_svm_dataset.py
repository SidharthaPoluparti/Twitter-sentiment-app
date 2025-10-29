import pandas as pd

# Load the original dataset that contains all classes
df = pd.read_csv('twitter_sentiment.csv')

# Keep only 'Positive', 'Negative', 'Neutral'
df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

# Remove empty 'clean_text' rows if any
df = df[df['clean_text'].notnull()]
df = df[df['clean_text'].str.strip() != '']

# Save the filtered dataset
df.to_csv('final_twitter_sentiment_svm.csv', index=False)

print("Filtered dataset with 3 classes saved as 'final_twitter_sentiment.csv'")
print("Class distribution:")
print(df['sentiment'].value_counts())
