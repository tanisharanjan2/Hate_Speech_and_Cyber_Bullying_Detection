import requests

url = "http://127.0.0.1:5000/predict"
data = {"text": "I hate you!"}

response = requests.post(url, json=data)

print("Response:", response.json())
