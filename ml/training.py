import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
TARGET = "species"
SHARED_DIR = Path(os.getenv("SHARED_DATA_DIR", "data"))

MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
}


def run(model_name: str):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODELS)}")

    df = pd.read_csv(SHARED_DIR / "processed.csv")
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MODELS[model_name]
    model.fit(X_train, y_train)

    acc = round(float(accuracy_score(y_test, model.predict(X_test))), 4)

    model_path = SHARED_DIR / f"candidate_{model_name}.joblib"
    metrics_path = SHARED_DIR / f"candidate_{model_name}.json"

    joblib.dump(model, model_path)
    metrics = {
        "model_name": model_name,
        "model_path": str(model_path),
        "accuracy": acc,
        "n_rows": len(df),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Trained {model_name}: {metrics}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "random_forest"
    run(name)
