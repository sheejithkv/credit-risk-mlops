from __future__ import annotations

import json
from io import StringIO

import joblib
import pandas as pd


def model_fn(model_dir: str):
    return joblib.load(f"{model_dir}/model.joblib")


def input_fn(request_body: str, request_content_type: str):
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        return pd.DataFrame([payload]) if isinstance(payload, dict) else pd.DataFrame(payload)
    if request_content_type == "text/csv":
        return pd.read_csv(StringIO(request_body))
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]
    return [
        {
            "prediction": int(prediction),
            "risk_label": "bad" if int(prediction) == 1 else "good",
            "probability_bad": float(probability),
        }
        for prediction, probability in zip(predictions, probabilities)
    ]


def output_fn(prediction, accept: str):
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
