from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import joblib
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ===============================
# Flask App
# ===============================

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*":{
            "origins":"*"
        }
    }
)


# ===============================
# NLTK Resources
# ===============================

try:
    stop_words = set(stopwords.words("english"))

except LookupError:

    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

    stop_words = set(stopwords.words("english"))


# ===============================
# Paths
# ===============================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "news_model.pkl"
)


# ===============================
# Load Model
# ===============================

print("Loading Model...")

model = joblib.load(MODEL_PATH)

print("Model Loaded Successfully")


# ===============================
# Text Cleaning
# ===============================

lemmatizer = WordNetLemmatizer()

stop_words = set(
    stopwords.words("english")
)


def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z ]",
        " ",
        text
    )

    words = text.split()


    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]


    return " ".join(words)



# ===============================
# Home Route
# ===============================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message":
        "News Category Classification API Running 🚀"
    })



# ===============================
# Prediction API
# ===============================
@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()


        if not data or "news" not in data:

            return jsonify({

                "success":False,

                "error":"News text is required"

            }),400



        news_text = data["news"]



        cleaned_text = clean_text(news_text)



        prediction = model.predict(
            [cleaned_text]
        )[0]



        probabilities = {}



        if hasattr(model,"predict_proba"):


            probs = model.predict_proba(
                [cleaned_text]
            )[0]


            classes = model.classes_


            for cls,prob in zip(classes,probs):

                probabilities[cls] = round(
                    float(prob*100),
                    2
                )



            confidence = round(
                max(probs)*100,
                2
            )


        else:

            confidence = None




        return jsonify({

            "success":True,

            "news":news_text,

            "category":prediction,

            "confidence":confidence,

            "probabilities":probabilities

        })



    except Exception as e:


        return jsonify({

            "success":False,

            "error":str(e)

        }),500



# ===============================
# Run Server
# ===============================

if __name__ == "__main__":

    port = int(os.environ.get("PORT",5001))

    app.run(
        host="0.0.0.0",
        port=port
    )