# ==========================================
# TASK-4: SENTIMENT ANALYSIS USING NLP
# CODTECH INTERNSHIP
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Download stopwords
nltk.download('stopwords')

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("IMDB_Dataset.csv")

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)

# ==========================================
# Data Preprocessing
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

# Convert sentiment labels
df['sentiment'] = df['sentiment'].map({
    'positive': 1,
    'negative': 0
})

# ==========================================
# Text Cleaning Function
# ==========================================

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df['clean_review'] = df['review'].apply(clean_text)

# ==========================================
# Feature Selection
# ==========================================

X = df['clean_review']
y = df['sentiment']

# ==========================================
# TF-IDF Vectorization
# ==========================================

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# Model Training
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ==========================================
# Model Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Model Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ==========================================
# Sentiment Distribution
# ==========================================

plt.figure(figsize=(6, 4))

sns.countplot(
    x=df['sentiment']
)

plt.title("Sentiment Distribution")
plt.xticks([0, 1], ["Negative", "Positive"])

plt.show()

# ==========================================
# Custom Prediction
# ==========================================

sample_review = [
    "This movie was amazing and I loved every moment of it."
]

sample_vector = vectorizer.transform(sample_review)

prediction = model.predict(sample_vector)

print("\nCustom Review Prediction:")

if prediction[0] == 1:
    print("Positive Sentiment")
else:
    print("Negative Sentiment")

# ==========================================
# Insights
# ==========================================

print("\n========== INSIGHTS ==========")

print("1. Successfully performed sentiment analysis using NLP techniques.")
print("2. Text data was cleaned and preprocessed for analysis.")
print("3. TF-IDF vectorization converted text into numerical features.")
print("4. Logistic Regression was used for sentiment classification.")
print("5. The model achieved high accuracy on movie reviews.")
print("6. Positive and negative sentiments were effectively identified.")
print("7. NLP techniques helped extract meaningful insights from textual data.")
