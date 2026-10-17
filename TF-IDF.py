import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PreprocessText import preprocess_text

train_df = pd.read_csv('train_set.csv')
test_df  = pd.read_csv('test_set.csv')

print(f"Train: {len(train_df)}  Test: {len(test_df)}")

MODES = ['tokenized', 'stemmed', 'lemmatized']

preprocessed = {}
for mode in MODES:
    print(f"Preprocessing ({mode})...")
    preprocessed[mode] = {
        'train': train_df['text'].apply(lambda x: preprocess_text(x, mode=mode)),
        'test':  test_df['text'].apply(lambda x: preprocess_text(x, mode=mode)),
    }


def top10_similar_pairs(texts, label=""):
    """Find top-10 unique most-similar tweet pairs using TF-IDF + cosine sim."""
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    mat = vec.fit_transform(texts)
    sim = cosine_similarity(mat)
    np.fill_diagonal(sim, -1)

    flat   = np.argsort(sim.ravel())[::-1]
    print(flat)
    pairs  = []
    seen   = set()
    texts_list = list(texts)

    for idx in flat:
        i, j = divmod(idx, sim.shape[1])
        key  = tuple(sorted([i, j]))
        if key not in seen:
            seen.add(key)
            if sim[i, j] >= 0.999:  # Skip literal identical copies
                continue
            pairs.append((i, j, sim[i, j]))
        if len(pairs) >= 10:
            break

    print(f"\n--- Top-10 Most Similar Tweet Pairs [{label}] ---")
    for rank, (i, j, score) in enumerate(pairs, 1):
        print(f"\n{rank}. Score: {score:.4f}")
        print(f"   A: {texts_list[i][:120]}")
        print(f"   B: {texts_list[j][:120]}")
    return pairs

for mode in MODES:
    top10_similar_pairs(preprocessed[mode]['test'], label=mode)

vectorizers = {
    'bow_binary': CountVectorizer(max_features=10000, ngram_range=(1, 2), binary=True),
    'bow_count':  CountVectorizer(max_features=10000, ngram_range=(1, 2)),
    'tfidf':      TfidfVectorizer(max_features=10000, ngram_range=(1, 2)),
}

X_train_text = preprocessed['lemmatized']['train']
X_test_text  = preprocessed['lemmatized']['test']

print("\n--- Vectorisation shapes ---")
feature_sets = {}
for name, vec in vectorizers.items():
    X_tr = vec.fit_transform(X_train_text)
    X_te = vec.transform(X_test_text)
    feature_sets[name] = {'train': X_tr, 'test': X_te}
    print(f"  {name}: train={X_tr.shape}  test={X_te.shape}")

import scipy.sparse as sp

sp.save_npz('X_train_tfidf.npz', feature_sets['tfidf']['train'])
sp.save_npz('X_test_tfidf.npz',  feature_sets['tfidf']['test'])
sp.save_npz('X_train_bow.npz',   feature_sets['bow_count']['train'])
sp.save_npz('X_test_bow.npz',    feature_sets['bow_count']['test'])

train_df['label'].to_csv('y_train.csv', index=False)
test_df['label'].to_csv('y_test.csv',   index=False)

print("\nAll feature matrices saved.")

