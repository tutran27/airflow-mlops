import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import requests
from airflow.sdk import dag, task
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

SHARED_DIR = Path(os.getenv("SHARED_DATA_DIR", "/opt/shared"))
ML_API_URL = os.getenv("ML_API_URL", "http://ml-api:8000")
MODEL_NAMES = ["random_forest", "decision_tree", "logistic_regression"]

DOCKER_OPTS = dict(
    image="airflow-mlops-ml:latest",
    docker_url="unix://var/run/docker.sock",
    network_mode="airflow_mlops_net",
    auto_remove="force",
    mount_tmp_dir=False,
    user=f"{os.getuid()}:0",
    environment={"SHARED_DATA_DIR": "/opt/shared"},
    mounts=[Mount(target="/opt/shared", source="airflow_mlops_shared", type="volume")],
)


@dag(
    dag_id="penguins_ml_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="*/5 * * * *",
    catchup=False,
)
def penguins_ml_pipeline():
    # Bước 1: Xử lý dữ liệu trong container ML
    process_data = DockerOperator(
        task_id="process_data",
        command="python -m ml.data_processing",
        **DOCKER_OPTS,
    )

    # Bước 2: Train song song 3 model
    train_tasks = [
        DockerOperator(
            task_id=f"train_{name}",
            command=f"python -m ml.training {name}",
            **DOCKER_OPTS,
        )
        for name in MODEL_NAMES
    ]

    # Bước 3: Chọn model tốt nhất & so sánh với model production hiện tại
    @task.branch
    def check_model() -> str:
        candidates = [
            json.loads((SHARED_DIR / f"candidate_{name}.json").read_text())
            for name in MODEL_NAMES
        ]
        best = max(candidates, key=lambda m: m["accuracy"])
        (SHARED_DIR / "best_candidate.json").write_text(json.dumps(best, indent=2))

        prod_path = SHARED_DIR / "prod_metrics.json"
        prod_acc = json.loads(prod_path.read_text())["accuracy"] if prod_path.exists() else -1

        print(f"All candidates: {[(c['model_name'], c['accuracy']) for c in candidates]}")
        print(f"Best={best['model_name']} ({best['accuracy']}) vs Prod={prod_acc}")

        if best["accuracy"] >= prod_acc:
            return "promote_and_reload"
        return "skip_deploy"

    # Bước 4a: Copy sang prod_* và gọi API reload
    @task
    def promote_and_reload():
        best = json.loads((SHARED_DIR / "best_candidate.json").read_text())
        shutil.copy2(best["model_path"], SHARED_DIR / "prod_model.joblib")
        shutil.copy2(SHARED_DIR / "best_candidate.json", SHARED_DIR / "prod_metrics.json")

        res = requests.post(f"{ML_API_URL}/reload", timeout=30)
        res.raise_for_status()
        print("API reloaded:", res.json())

    # Bước 4b: Giữ nguyên model cũ
    @task
    def skip_deploy():
        print("Model mới không tốt hơn, giữ nguyên model cũ.")

    process_data >> train_tasks >> check_model() >> [promote_and_reload(), skip_deploy()]


penguins_ml_pipeline()