import os
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===============================
# Download NLTK Resources
# ===============================

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ===============================
# Paths
# ===============================
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "news_model.pkl"
)

TFIDF_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "tfidf.pkl"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "label_encoder.pkl"
)

# ===============================
# Load Model
# ===============================

model = joblib.load(MODEL_PATH)

# ===============================
# Text Preprocessing
# ===============================

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return ' '.join(words)

# ===============================
# Prediction Function
# ===============================

def predict_news(text):

    cleaned_text = clean_text(text)

    prediction = model.predict([cleaned_text])[0]

    probabilities = model.predict_proba([cleaned_text])[0]

    confidence = round(max(probabilities) * 100, 2)

    class_probabilities = {}

    for label, prob in zip(model.classes_, probabilities):
        class_probabilities[label] = round(prob * 100, 2)

    return {
        'text': text,
        'cleaned_text': cleaned_text,
        'category': prediction,
        'confidence': confidence,
        'probabilities': class_probabilities
    }

# ===============================
# Test
# ===============================

if __name__ == '__main__':

    sample_news = 'India won the cricket world cup after a thrilling final match.'

    result = predict_news(sample_news)

    print('\\n===== Prediction Result =====')

    print(f'Text       : {result["text"]}')

    print(f'Category   : {result["category"]}')

    print(f'Confidence : {result["confidence"]}%')

    print('\\nProbabilities:')

    for category, score in result['probabilities'].items():
        print(f'{category} : {score}%')