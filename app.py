from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "credit_card_approval_model.joblib"

app = Flask(__name__)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run `python train_model.py` before starting Flask."
        )
    return joblib.load(MODEL_PATH)


model_bundle = load_model()
FEATURES = model_bundle["features"]
MODEL_NAME = model_bundle.get("model_name", "Trained classifier")


@app.get("/")
def index():
    return render_template("index.html", features=FEATURES, model_name=MODEL_NAME)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    missing = [field for field in FEATURES if field not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    row = pd.DataFrame([{field: payload[field] for field in FEATURES}])
    prediction = int(model_bundle["pipeline"].predict(row)[0])

    probability = None
    if hasattr(model_bundle["pipeline"], "predict_proba"):
        probability = float(model_bundle["pipeline"].predict_proba(row)[0][prediction])

    return jsonify(
        {
            "prediction": "Approved" if prediction == 1 else "Rejected",
            "decision_code": prediction,
            "confidence": probability,
            "model": MODEL_NAME,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
