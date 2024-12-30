from flask import Flask, request, jsonify, render_template
import joblib
import re
import numpy as np
import scipy.sparse as sp

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model, vectorizer, and label encoder
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML page

# API route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON input
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "Text field is missing"}), 400

    # Step 1: Clean input text
    cleaned_text = clean_text(data['text'])

    # Step 2: Extract TF-IDF features
    tfidf_features = vectorizer.transform([cleaned_text])

    # Step 3: Extract handcrafted features
    word_count = len(cleaned_text.split())  # Number of words
    char_count = len(cleaned_text)          # Number of characters
    contains_economy = 1 if 'اقتصاد' in cleaned_text else 0  # Check for specific keyword

    # Combine handcrafted features
    handcrafted_features = np.array([[word_count, char_count, contains_economy]])

    # Step 4: Combine TF-IDF and handcrafted features
    full_features = sp.hstack([tfidf_features, sp.csr_matrix(handcrafted_features)])

    # Step 5: Predict class
    prediction = model.predict(full_features)
    predicted_class = label_encoder.inverse_transform(prediction)[0]

    # Step 6: Return prediction
    return jsonify({"text": data['text'], "predicted_category": predicted_class})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
