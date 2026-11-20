import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SHARED_DIR = Path(os.getenv("SHARED_DATA_DIR", "data"))
MODEL_PATH = SHARED_DIR / "prod_model.joblib"
META_PATH = SHARED_DIR / "prod_metrics.json"

app = FastAPI(title="Penguins API")
model = None
metrics = None


class PenguinInput(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float


def load_model() -> bool:
    global model, metrics
    if not MODEL_PATH.exists():
        return False
    model = joblib.load(MODEL_PATH)
    metrics = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    return True


@app.on_event("startup")
def startup():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "metrics": metrics}


@app.post("/reload")
def reload_model():
    if not load_model():
        raise HTTPException(status_code=404, detail="Chưa có prod_model.joblib")
    return {"status": "reloaded", "metrics": metrics}


@app.post("/predict")
def predict(data: PenguinInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Chưa có model, hãy chạy DAG train trước")
    df = pd.DataFrame([data.model_dump()])
    species = str(model.predict(df)[0])
    return {"species": species, "metrics": metrics}
