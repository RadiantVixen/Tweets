import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X_train_tfidf = pd.read_csv('X_train_tfidf.csv')
y_train = pd.read_csv('y_train.csv')
X_test_tfidf = pd.read_csv('X_test_tfidf.csv')
y_test = pd.read_csv('y_test.csv')


# Initialize the classifier
model = LogisticRegression(max_iter=1000, random_state=42)

# Train (fit) the model on the TF-IDF training data
model.fit(X_train_tfidf, y_train)

# Predict on the test data
y_pred = model.predict(X_test_tfidf)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Detailed report per class
print(classification_report(y_test, y_pred))