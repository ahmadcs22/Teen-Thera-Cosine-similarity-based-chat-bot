# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re

app = Flask(__name__)
CORS(app)

# Load compressed resources
vectorizer = joblib.load('vectorizer_compressed.joblib')
X = joblib.load('tfidf_matrix_compressed.joblib')
questions_list = joblib.load('questions_list.joblib')
answers_list = joblib.load('answers_list.joblib')

# Text preprocessing
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_with_stopwords(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed = [stemmer.stem(token) for token in lemmatized]
    return ' '.join(stemmed)

@app.route('/', methods=['GET'])
def home():
    return "Chatbot API is running!"

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    processed = preprocess_with_stopwords(user_input)

    vec = vectorizer.transform([processed])
    similarities = cosine_similarity(vec, X)
    max_sim = np.max(similarities)

    if max_sim < 0.6:
        return jsonify({'response': "I can't answer this question."})

    best_idx = np.argmax(similarities)
    return jsonify({'response': answers_list[best_idx]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
