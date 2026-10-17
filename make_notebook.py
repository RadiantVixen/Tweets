import nbformat as nbf

nb = nbf.v4.new_notebook()

def md(src):   return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

nb.cells = [

md("""# 🐦 Project 00 – Tweets
## Intro to Natural Language Processing · Sentiment Analysis

This notebook covers:
1. **Data loading & cleaning**
2. **Text preprocessing** – tokenization, stemming, lemmatization, stop-words
3. **Vectorisation** – Bag-of-Words (binary & count) and TF-IDF with bigrams
4. **Cosine similarity** – top-10 most similar tweet pairs per preprocessing variant
5. **Machine learning** – multiple classifiers, comparison, accuracy targets
"""),

md("## 1 · Imports & Setup"),
code("""\
import re, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection         import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.linear_model            import LogisticRegression
from sklearn.naive_bayes             import ComplementNB
from sklearn.svm                     import LinearSVC
from sklearn.ensemble                import RandomForestClassifier
from sklearn.metrics                 import accuracy_score, classification_report

import nltk
from nltk.corpus   import stopwords
from nltk.stem     import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    nltk.download(pkg, quiet=True)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
print("All imports OK ✓")
"""),

md("""## 2 · Data Loading

The raw CSV files store tweets as comma-separated values on a single line.
We load each sentiment file and assign the appropriate label.
"""),
code("""\
def load_csv_tweets(filepath, label):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    tweets = [t.strip() for t in content.split(',') if t.strip()]
    return pd.DataFrame({'text': tweets, 'label': label})

df_neg = load_csv_tweets('data/processedNegative.csv', 'negative')
df_neu = load_csv_tweets('data/processedNeutral.csv',  'neutral')
df_pos = load_csv_tweets('data/processedPositive.csv', 'positive')

print(f"Negative: {len(df_neg):>5}  Neutral: {len(df_neu):>5}  Positive: {len(df_pos):>5}")
"""),

code("""\
df = pd.concat([df_neg, df_neu, df_pos], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].str.len() > 0]

print(f"Total tweets after cleaning: {len(df)}")
df['label'].value_counts().plot(kind='bar', color=['#e74c3c','#95a5a6','#2ecc71'],
                                edgecolor='black', figsize=(5,3))
plt.title('Class Distribution')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""),

md("## 3 · Train / Test Split (80 / 20, stratified)"),
code("""\
X, y = df['text'], df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = pd.DataFrame({'text': X_train, 'label': y_train}).reset_index(drop=True)
test_df  = pd.DataFrame({'text': X_test,  'label': y_test}).reset_index(drop=True)

print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")
print(f"\\nTest class distribution:\\n{y_test.value_counts()}")
"""),

md("""## 4 · Text Preprocessing

Three variants applied independently:

| Mode | What it does |
|------|-------------|
| `tokenized` | Tokenize + remove stop-words |
| `stemmed` | As above + Porter stemmer |
| `lemmatized` | As above + WordNet lemmatizer |

Tweets are also cleaned (URLs, @mentions, hashtags, punctuation removed) first.
"""),
code("""\
stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r'http\\S+|www\\S+', '', text)
    text = re.sub(r'@\\w+', '', text)
    text = re.sub(r'#(\\w+)', r'\\1', text)
    text = re.sub(r'[^a-z\\s]', '', text)
    return re.sub(r'\\s+', ' ', text).strip()

def preprocess_text(text, mode='tokenized'):
    tokens = word_tokenize(clean_tweet(text))
    out = []
    for t in tokens:
        if t in stop_words or not t.strip():
            continue
        if mode == 'stemmed':
            out.append(stemmer.stem(t))
        elif mode == 'lemmatized':
            out.append(lemmatizer.lemmatize(t))
        else:
            out.append(t)
    return ' '.join(out)

sample = train_df['text'].iloc[0]
print(f"Original   : {sample[:100]}")
for m in ['tokenized', 'stemmed', 'lemmatized']:
    print(f"{m:12}: {preprocess_text(sample, m)[:100]}")
"""),

code("""\
MODES = ['tokenized', 'stemmed', 'lemmatized']
preprocessed = {}
for mode in MODES:
    print(f"Preprocessing ({mode})...", end=' ')
    preprocessed[mode] = {
        'train': train_df['text'].apply(lambda x: preprocess_text(x, mode=mode)),
        'test':  test_df['text'].apply(lambda x: preprocess_text(x, mode=mode)),
    }
    print("done ✓")
"""),

md("""## 5 · Cosine Similarity – Top-10 Most Similar Tweet Pairs

Computed separately for each preprocessing variant using TF-IDF vectors.
"""),
code("""\
def top10_similar_pairs(texts, label=""):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    mat = vec.fit_transform(texts)
    sim = cosine_similarity(mat)
    np.fill_diagonal(sim, -1)

    flat, pairs, seen = np.argsort(sim.ravel())[::-1], [], set()
    texts_list = list(texts)
    for idx in flat:
        i, j = divmod(idx, sim.shape[1])
        key  = tuple(sorted([i, j]))
        if key not in seen:
            seen.add(key)
            pairs.append((i, j, sim[i, j]))
        if len(pairs) >= 10:
            break

    print(f"\\n{'─'*60}\\n  Top-10 Similar Pairs [{label}]\\n{'─'*60}")
    for rank, (i, j, score) in enumerate(pairs, 1):
        print(f"\\n{rank:>2}. Score: {score:.4f}")
        print(f"    A: {texts_list[i][:110]}")
        print(f"    B: {texts_list[j][:110]}")
    return pairs

for mode in MODES:
    top10_similar_pairs(preprocessed[mode]['test'], label=mode)
"""),

md("""## 6 · Vectorisation

| Name | Description |
|------|-------------|
| **BoW binary** | 1 if word present, 0 otherwise |
| **BoW count** | Raw word counts |
| **TF-IDF** | Term frequency × inverse document frequency |

All use bigrams (`ngram_range=(1,2)`) and top 10 000 features.
"""),
code("""\
vectorizers = {
    'bow_binary': CountVectorizer(max_features=10000, ngram_range=(1,2), binary=True),
    'bow_count':  CountVectorizer(max_features=10000, ngram_range=(1,2)),
    'tfidf':      TfidfVectorizer(max_features=10000, ngram_range=(1,2)),
}

X_tr_text = preprocessed['lemmatized']['train']
X_te_text  = preprocessed['lemmatized']['test']

feature_sets = {}
print(f"{'Name':<15}  Train shape       Test shape")
print("─"*45)
for name, vec in vectorizers.items():
    tr = vec.fit_transform(X_tr_text)
    te = vec.transform(X_te_text)
    feature_sets[name] = {'train': tr, 'test': te}
    print(f"{name:<15}  {str(tr.shape):<18} {te.shape}")
"""),

md("""## 7 · Machine Learning – Sentiment Classification

Four classifiers trained on both TF-IDF and BoW (count) features.

| Target | Threshold |
|--------|-----------|
| Mandatory | ≥ 0.832 |
| Bonus 1   | ≥ 0.851 |
| Bonus 2   | ≥ 0.873 |
"""),
code("""\
classifiers = {
    'Logistic Regression':    LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42),
    'Complement Naive Bayes': ComplementNB(alpha=0.1),
    'Linear SVM':             LinearSVC(max_iter=2000, C=0.5, random_state=42),
    'Random Forest':          RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}

THRESHOLDS = [(0.873,'🏆 BONUS ≥0.873'),(0.851,'⭐ BONUS ≥0.851'),
              (0.832,'✅ MANDATORY'),   (0,'❌ below target')]

def tier(acc):
    for t, label in THRESHOLDS:
        if acc >= t: return label

results = []
for feat_name, fkey in [('TF-IDF','tfidf'),('BoW (count)','bow_count')]:
    X_tr = feature_sets[fkey]['train']
    X_te = feature_sets[fkey]['test']
    print(f"\\n{'═'*60}\\n  Feature set: {feat_name}\\n{'═'*60}")
    for clf_name, clf in classifiers.items():
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        acc    = accuracy_score(y_test, y_pred)
        t      = tier(acc)
        print(f"\\n[{feat_name}] {clf_name}  →  {acc:.4f}  {t}")
        print(classification_report(y_test, y_pred, digits=4))
        results.append({'Features': feat_name, 'Classifier': clf_name, 'Accuracy': acc, 'Tier': t})
"""),

md("## 8 · Summary & Visualisation"),
code("""\
summary = pd.DataFrame(results).sort_values('Accuracy', ascending=False).reset_index(drop=True)
display(summary)
"""),
code("""\
fig, ax = plt.subplots(figsize=(10, 5))
colors  = ['#27ae60' if a >= 0.873 else '#f39c12' if a >= 0.832 else '#e74c3c'
           for a in summary['Accuracy']]
labels  = [f"{r['Features']} · {r['Classifier']}" for _, r in summary.iterrows()]
bars    = ax.barh(labels, summary['Accuracy'], color=colors, edgecolor='white')
ax.axvline(0.832, color='orange', linestyle='--', linewidth=1.5, label='Mandatory 0.832')
ax.axvline(0.873, color='green',  linestyle='--', linewidth=1.5, label='Bonus 0.873')
ax.set_xlabel('Accuracy')
ax.set_title('Classifier Accuracy Comparison')
ax.legend()
for bar, val in zip(bars, summary['Accuracy']):
    ax.text(bar.get_width()-0.002, bar.get_y()+bar.get_height()/2,
            f'{val:.4f}', va='center', ha='right', color='white', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.show()
"""),
code("""\
best = summary.iloc[0]
print(f"Best model : [{best['Features']}] {best['Classifier']}")
print(f"Accuracy   : {best['Accuracy']:.4f}  {best['Tier']}")
"""),

md("---\n*Project 00 – Tweets · NLP Sentiment Analysis*"),
]

import json
output = 'tweets_sentiment_analysis.ipynb'
with open(output, 'w') as f:
    nbf.write(nb, f)
print(f"Notebook written → {output}")
