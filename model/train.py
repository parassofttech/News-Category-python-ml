import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
import re

# ===============================
# Download NLTK Resources
# ===============================

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ===============================
# Paths
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "bbc-news.csv")

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")
LABEL_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# ===============================
# Load Dataset
# ===============================

print("Loading Dataset...")

df = pd.read_csv(DATASET_PATH)

df = df.dropna()

category_count = df["category"].value_counts()

valid_categories = category_count[category_count >= 2].index

df = df[df["category"].isin(valid_categories)]

print(f"Total Records : {len(df)}")

print(df["category"].value_counts())

# ===============================
# Text Cleaning
# ===============================

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\\S+", "", text)

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

print("Cleaning Text...")

df["clean_text"] = df["text"].astype(str).apply(clean_text)

# ===============================
# Features
# ===============================

X = df["clean_text"]

y = df["category"]

# ===============================
# Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ===============================
# TF-IDF
# ===============================

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,3),
    sublinear_tf=True
)

# ===============================
# Model
# ===============================

classifier = MultinomialNB(
    alpha=0.1
)

pipeline = Pipeline([
    ("tfidf", vectorizer),
    ("model", classifier)
])

print("Training Model...")

pipeline.fit(X_train, y_train)

# ===============================
# Prediction
# ===============================

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("Model Accuracy")
print("==============================")
print(f"{accuracy*100:.2f}%")

print("\n==============================")
print("Classification Report")
print("==============================")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# ===============================
# Save Model
# ===============================

joblib.dump(pipeline, MODEL_PATH)

joblib.dump(pipeline.named_steps["tfidf"], VECTORIZER_PATH)

joblib.dump(sorted(df["category"].unique()), LABEL_PATH)

print("\nModel Saved Successfully")

print(MODEL_PATH)
print(VECTORIZER_PATH)
print(LABEL_PATH)