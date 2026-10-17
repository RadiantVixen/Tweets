import pandas as pd
from sklearn.model_selection import train_test_split

def load_csv_tweets(filepath, label):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    tweets = [t.strip() for t in content.split(',') if t.strip()]
    print(tweets)
    return pd.DataFrame({'text': tweets, 'label': label})


df_neg = load_csv_tweets('data/processedNegative.csv', 'negative')
df_neu = load_csv_tweets('data/processedNeutral.csv', 'neutral')
df_pos = load_csv_tweets('data/processedPositive.csv', 'positive')

print(f"Negative: {len(df_neg)} | Neutral: {len(df_neu)} | Positive: {len(df_pos)}")

df = pd.concat([df_neg, df_neu, df_pos], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].str.len() > 0]

print(f"Total samples after cleaning: {len(df)}")
print(df['label'].value_counts())

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = pd.DataFrame({'text': X_train, 'label': y_train})
test_df  = pd.DataFrame({'text': X_test,  'label': y_test})

train_df.to_csv('train_set.csv', index=False)
test_df.to_csv('test_set.csv', index=False)

# print(f"\nSaved: train={len(train_df)}  test={len(test_df)}")
