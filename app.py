import os

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from transformers import pipeline

load_dotenv()

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "distilbert-base-uncased-finetuned-sst-2-english"
)
HF_TOKEN = os.getenv("HF_TOKEN")
APP_PORT = int(os.getenv("APP_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

app = Flask(__name__)

pipeline_kwargs = {
    "task": "sentiment-analysis",
    "model": MODEL_NAME
}

if HF_TOKEN:
    pipeline_kwargs["token"] = HF_TOKEN

clf = pipeline(**pipeline_kwargs)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text")

    if not text or not isinstance(text, str):
        return jsonify({"error": "Invalid input. Provide JSON: {\"text\": \"...\"}"}), 400

    result = clf(text)[0]

    return jsonify({
        "input": text,
        "label": result["label"],
        "score": float(result["score"])
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=APP_PORT, debug=FLASK_DEBUG)