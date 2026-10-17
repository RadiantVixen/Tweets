import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model    import LogisticRegression
from sklearn.naive_bayes     import MultinomialNB, ComplementNB
from sklearn.svm             import LinearSVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score, classification_report

X_train_tfidf = sp.load_npz('X_train_tfidf.npz')
X_test_tfidf  = sp.load_npz('X_test_tfidf.npz')
X_train_bow   = sp.load_npz('X_train_bow.npz')
X_test_bow    = sp.load_npz('X_test_bow.npz')

y_train = pd.read_csv('y_train.csv')['label']
y_test  = pd.read_csv('y_test.csv')['label']

print(f"Train: {X_train_tfidf.shape}  Test: {X_test_tfidf.shape}")
print(f"Class distribution (test):\n{y_test.value_counts()}\n")

classifiers = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs', random_state=42
    ),
    'Complement Naive Bayes': ComplementNB(alpha=0.1),
    'Linear SVM':            LinearSVC(max_iter=2000, C=0.5, random_state=42),
    'Random Forest':         RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    ),
}

MANDATORY_THRESHOLD = 0.832
BONUS_1             = 0.851
BONUS_2             = 0.873

results = []

for feat_name, (X_tr, X_te) in [('TF-IDF', (X_train_tfidf, X_test_tfidf)),
                                  ('BoW',    (X_train_bow,   X_test_bow))]:
    print(f"\n{'='*60}")
    print(f"  Feature set: {feat_name}")
    print(f"{'='*60}")

    for clf_name, clf in classifiers.items():
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        acc    = accuracy_score(y_test, y_pred)

        if acc >= BONUS_2:
            tier = "🏆 BONUS ≥0.873"
        elif acc >= BONUS_1:
            tier = "⭐ BONUS ≥0.851"
        elif acc >= MANDATORY_THRESHOLD:
            tier = "✅ MANDATORY"
        else:
            tier = "❌ below target"

        print(f"\n[{feat_name}] {clf_name}")
        print(f"  Accuracy: {acc:.4f}  {tier}")
        print(classification_report(y_test, y_pred, digits=4))

        results.append({
            'features':   feat_name,
            'classifier': clf_name,
            'accuracy':   acc,
            'tier':       tier,
        })


print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
summary = pd.DataFrame(results).sort_values('accuracy', ascending=False)
print(summary[['features', 'classifier', 'accuracy', 'tier']].to_string(index=False))

best = summary.iloc[0]
print(f"\nBest: [{best['features']}] {best['classifier']} → {best['accuracy']:.4f}  {best['tier']}")
