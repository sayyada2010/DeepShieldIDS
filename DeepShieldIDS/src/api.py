
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from .model import AttentionPooling
from .config import MODEL_DIR

MODEL_PATH = MODEL_DIR / "hybrididsnet_cic.keras"
app = FastAPI(title="DeepShieldIDS API", version="1.0")
model = None

class TrafficRequest(BaseModel):
    features: List[float]

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"AttentionPooling": AttentionPooling})

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(req: TrafficRequest):
    if model is None:
        return {"error": f"Model not found at {MODEL_PATH}. Train first or update MODEL_PATH."}
    x = np.asarray(req.features, dtype="float32").reshape(1, -1, 1)
    prob = float(model.predict(x, verbose=0).ravel()[0])
    label = "attack" if prob >= 0.5 else "normal"
    return {"prediction": label, "confidence": prob if label == "attack" else 1 - prob, "attack_probability": prob, "alert": label == "attack"}

@app.get("/model-info")
def model_info():
    if model is None:
        return {"model_loaded": False}
    return {"model_loaded": True, "input_shape": model.input_shape, "output_shape": model.output_shape}
