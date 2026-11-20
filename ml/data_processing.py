import os
from pathlib import Path

import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
NUM_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
SHARED_DIR = Path(os.getenv("SHARED_DATA_DIR", "data"))


def run():
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = SHARED_DIR / "raw.csv"
    processed_path = SHARED_DIR / "processed.csv"

    # Mock data mới: mỗi lần chạy lấy ngẫu nhiên 100 dòng cộng dồn vào raw.csv
    batch = pd.read_csv(DATA_URL).sample(n=100, replace=True)
    batch.to_csv(raw_path, mode="a", header=not raw_path.exists(), index=False)

    # Làm sạch giống hệt trong notebook
    df = pd.read_csv(raw_path)
    df[NUM_COLS] = df[NUM_COLS].fillna(df[NUM_COLS].mean())
    df = df.dropna(subset=["sex"])

    df.to_csv(processed_path, index=False)
    print(f"Done processing: {len(df)} rows -> {processed_path}")


if __name__ == "__main__":
    run()
