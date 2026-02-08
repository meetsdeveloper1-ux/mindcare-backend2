import requests
from flask import Flask, request, jsonify
import os
from langdetect import detect
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
PRIVATE_TOKEN = os.getenv("PRIVATE_TOKEN")

@app.route("/chat", methods=["POST"])
def chat():
    token = request.headers.get("Authorization")
    if token != PRIVATE_TOKEN:
        return jsonify({"error": "Access denied"}), 403

    if not request.json or "message" not in request.json:
        return jsonify({"error": "Missing message field"}), 400

    text = request.json["message"]
    try:
        language = detect(text)
    except Exception as e:
        return jsonify({"error": f"Language detection failed: {str(e)}"}), 400

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text}

    try:
        res = requests.post(
            "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill",
            headers=headers,
            json=payload
        )
        res.raise_for_status()
        reply = res.json()[0]["generated_text"]
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"API request failed: {str(e)}"}), 500
    except (KeyError, IndexError) as e:
        return jsonify({"error": f"Invalid API response: {str(e)}"}), 500

    return jsonify({
        "reply": reply,
        "language": language
    })

app.run(debug=True)
