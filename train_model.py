# train_model.py (Updated for separate Fake and True CSVs)

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from preprocess import clean_text

# Load both CSV files
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

# Add labels
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# Combine both
df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

# Use only relevant columns (if needed)
df['text'] = df['text'].fillna('')
df['cleaned_text'] = df['text'].apply(clean_text)

# Features & labels
X = df['cleaned_text']
y = df['label'].map({'FAKE': 0, 'REAL': 1})

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_vec = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'models/logistic_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
