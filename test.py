from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Flask is running!"

if __name__ == "__main__":
    print("🚀 Flask server starting...")
    app.run(debug=True)
