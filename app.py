from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Pretrained model (fast + common). Downloads on first run.
# This model outputs POSITIVE / NEGATIVE with a score.
clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text")

    if not text or not isinstance(text, str):
        return jsonify({"error": "Invalid input. Provide JSON: {\"text\": \"...\"}"}), 400

    result = clf(text)[0]  # e.g. {"label": "POSITIVE", "score": 0.999...}
    return jsonify({
        "input": text,
        "label": result["label"],
        "score": float(result["score"])
    })

if __name__ == "__main__":
    # Local dev run (not used in Docker; Docker uses waitress)
    app.run(host="0.0.0.0", port=5000, debug=True)