import pandas as pd

# Load dataset
df = pd.read_csv("labeled_data.csv")  # Ensure the correct filename

# Display first few rows to verify
print(df.head())

print(df.columns)
df = df.rename(columns={"tweet": "Text", "class": "Label"})
df = df.drop(columns=["Unnamed: 0"])
print(df.head())
print(df.info())  # Overview of dataset
print(df.describe())  # Statistical summary
print(df["Label"].value_counts())
#0 - hate speech
#1 - offensive language
#2 - neither
print(df.isnull().sum())
df.to_csv("cleaned_dataset.csv", index=False)
df["Text"] = df["Text"].str.lower()
import re

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"\@\w+|\#", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    return text

df["Text"] = df["Text"].apply(clean_text)
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

df["Text"] = df["Text"].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()

df["Text"] = df["Text"].apply(lambda x: [word.lower() for word in x] if isinstance(x, list) else x.lower())

df.to_csv("cleaned_dataset.csv", index=False)
print(df.isnull().sum())  # Should print 0
df.reset_index(drop=True, inplace=True)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df["Text"])
import joblib
# Save the trained TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
print("Vectorizer saved successfully!")
print("X_tfidf shape:", X_tfidf.shape)  # Should be (24783, 21635)
print("df shape:", df.shape)  # Should be (24783,)
print(df.head())  # See what is inside y
y = df["Label"].values.ravel()  # Change "Label" to your actual target column name
print(df.shape)  # Should now be (24783,)
df["Label"] = df[["hate_speech", "offensive_language", "neither"]].idxmax(axis=1)
y = df["Label"]
print(y.shape)  # Should be (24783,)
# Convert to DataFrame
X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Print shape to check dimensions
print(X_tfidf_df.shape)
#X_tfidf_df.to_csv("tfidf_features.csv",index=False)
from sklearn.model_selection import train_test_split

# Features (X) - The numerical representation of text
X = X_tfidf  # or X_bow if you're using CountVectorizer

# Labels (y) - The target values (hate speech or not)
y = df["Label"]  # Assuming "Label" is the column with class labels (0/1)

# Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print dataset sizes
print("Training set size:", X_train.shape)
print("Testing set size:",X_test.shape)
from sklearn.linear_model import LogisticRegression

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

print("✅ Model training completed!")
from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"🔹 Accuracy: {accuracy:.4f}")

# Show detailed classification report
print(classification_report(y_test, y_pred))
import joblib

joblib.dump(model, "hate_speech_model.pkl")
print("✅ Model saved successfully!")
def predict_text(text, model, vectorizer):
    # Convert text into TF-IDF features
    text_tfidf = vectorizer.transform([text])

    # Make a prediction
    prediction = model.predict(text_tfidf)[0]

    return prediction

# Example test
new_text = "I hate you so much!"
predicted_label = predict_text(new_text, model, tfidf_vectorizer)
print(f"🔹 Predicted Label: {predicted_label}")
from flask import Flask, request, jsonify
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = joblib.load("hate_speech_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Ensure you saved this earlier

# Initialize Flask app
app = Flask(__name__)

# Define the API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text"]  # Get text input from request
    text_tfidf = vectorizer.transform([data])  # Transform text into TF-IDF features
    prediction = model.predict(text_tfidf)[0]  # Predict label
    
    return jsonify({"prediction": prediction})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

import joblib

# Load Model & Vectorizer
model = joblib.load("hate_speech_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Test with "Good" input
sample_text = ["I am writing "]
sample_features = vectorizer.transform(sample_text)
prediction = model.predict(sample_features)[0]
confidence = model.predict_proba(sample_features).max()

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2f}")











