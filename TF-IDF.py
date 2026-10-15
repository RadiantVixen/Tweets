import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PreprocessText import preprocess_text

# Load your train and test sets (from your saved CSVs)
train_df = pd.read_csv('train_set.csv')
test_df = pd.read_csv('test_set.csv')

# for text in train_df['text']:
#     preprocessed_text = preprocess_text(text, mode='tokenized')
#     print(preprocessed_text)
#     preprocessed_text = preprocess_text(text, mode='lemmatized')
#     print(preprocessed_text)
#     preprocessed_text = preprocess_text(text, mode='stemmed')
#     print(preprocessed_text)


# Initialize the TF-IDF Vectorizer
# max_features limits the vocabulary size, ngram_range captures words and bigrams
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# Fit and transform the training text into numerical matrices
X_train_tfidf = vectorizer.fit_transform(train_df['text'])
y_train = train_df['label']

# Transform the test text using the SAME fitted vectorizer
X_test_tfidf = vectorizer.transform(test_df['text'])
y_test = test_df['label'] 

print("TF-IDF Train Matrix Shape:", X_train_tfidf.shape)
print("TF-IDF Test Matrix Shape:", X_test_tfidf.shape)

df = pd.concat([train_df, test_df], ignore_index=True)

tweets = df['text'].tolist()

# 2. Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(tweets)

# 3. Calculate Cosine Similarity matrix across all tweets
print("Calculating cosine similarity matrix...")
similarity_matrix = cosine_similarity(X_test_tfidf, X_test_tfidf)

# 4. Extract top 10 most similar tweet pairs
# We set diagonal to -1 or 0 so a tweet doesn't match with itself
np.fill_diagonal(similarity_matrix, -1)

# Find indices of the top similarity scores
# We flatten the matrix and argsort to find highest scores
flat_indices = np.argsort(similarity_matrix.ravel())[::-1]

top_pairs = []
seen_pairs = set()

for idx in flat_indices:
    # Convert flat index back to 2D matrix coordinates (row, col)
    i, j = divmod(idx, similarity_matrix.shape[1])
    score = similarity_matrix[i, j]
    
    # Since similarity matrix is symmetric (A vs B is same as B vs A), avoid duplicates
    pair_key = tuple(sorted([i, j]))
    
    if pair_key not in seen_pairs:
        seen_pairs.add(pair_key)
        top_pairs.append((i, j, score))
        
    if len(top_pairs) >= 10:
        break

# 5. Display the top 10 most similar tweet pairs
print("\n--- Top-10 Most Similar Pairs of Tweets ---")
for rank, (i, j, score) in enumerate(top_pairs, 1):
    print(f"\n{rank}. Similarity Score: {score:.4f}")
    print(f"   - Tweet A: {tweets[i]}")
    print(f"   - Tweet B: {tweets[j]}")


column_names1 = [f"feature_{i}" for i in range(X_train_tfidf.shape[1])]
column_names2 = [f"feature_{i}" for i in range(y_train.shape[1])]
column_names3 = [f"feature_{i}" for i in range(X_test_tfidf.shape[1])]
column_names4 = [f"feature_{i}" for i in range(y_test.shape[1])]

X_train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=column_names1)
y_train_df = pd.DataFrame(y_train.toarray(), columns=column_names2)
X_test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=column_names3)
y_test_df = pd.DataFrame(y_test.toarray(), columns=column_names4)

X_train_df.to_csv('X_train_df.csv', index=False)
y_train_df.to_csv('y_train_df.csv', index=False)
X_test_df.to_csv('X_test_df.csv', index=False)
y_test_df.to_csv('y_test_df.csv', index=False)
