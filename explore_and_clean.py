import pandas as pd

column_names = ['tweet_id', 'entity', 'sentiment', 'text']
df = pd.read_csv('twitter_sentiment.csv', header=None, names=column_names)

# Remove 'Irrelevant' rows
df = df[df['sentiment'] != 'Irrelevant']

# Merge 'Neutral' into 'Negative'
df['sentiment'] = df['sentiment'].replace({'Neutral': 'Negative'})

print("Value counts after filtering:")
print(df['sentiment'].value_counts())

# Save cleaned data if you want
df.to_csv('cleaned_twitter_sentiment.csv', index=False)
