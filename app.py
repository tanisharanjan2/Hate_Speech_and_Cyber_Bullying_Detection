from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Load the trained model and vectorizer
model = joblib.load("hate_speech_model.pkl")  # Load trained model
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Load TF-IDF vectorizer

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("text", "")  # Get input text safely
        if not data:
            return jsonify({"error": "No input text provided"}), 400

        text_tfidf = vectorizer.transform([data])  # Convert text to TF-IDF features
        prediction = model.predict(text_tfidf)[0]  # Predict label
        confidence = model.predict_proba(text_tfidf).max()  # Get confidence score
        if confidence < 0.6:
          prediction = "Not Sure / Neutral" 
        return jsonify({"prediction": prediction, "confidence": round(confidence, 4)})

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    print("🚀 Flask server starting on http://127.0.0.1:5000 ...")
    app.run(debug=True)
