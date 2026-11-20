# Airflow MLOps Pipeline (Penguins Classification)

Hệ thống MLOps end-to-end sử dụng **Apache Airflow 3.4.0**, **Docker (`DockerOperator`)**, **Scikit-learn** và **FastAPI**: tự động lấy thêm dữ liệu mới định kỳ, làm sạch dữ liệu, huấn luyện song song 3 mô hình trong các container tách biệt, so sánh chất lượng với mô hình production hiện tại (Champion vs Challenger) và tự động hot-reload sang API phục vụ dự đoán.

---

## 1. Kiến trúc & Luồng hoạt động

### Luồng DAG (`penguins_ml_pipeline` — lịch chạy mặc định `*/5 * * * *`)
```text
                  ┌──► train_random_forest ──────┐                        ┌──► promote_and_reload ──(POST /reload)──► ml-api
process_data ─────┼──► train_decision_tree ──────┼──► check_model ────────┤
                  └──► train_logistic_regression ┘    (@task.branch)      └──► skip_deploy
```

1. **`process_data` (`DockerOperator`)**: Bật container ML (`airflow-mlops-ml:latest`), lấy ngẫu nhiên 100 dòng cộng dồn vào `raw.csv`, làm sạch (điền thiếu cột số bằng mean, bỏ dòng thiếu `sex`) và xuất ra `processed.csv`.
2. **`train_*` (`DockerOperator` x3 song song)**: Bật 3 container ML riêng cho `random_forest`, `decision_tree`, `logistic_regression`; mỗi container ghi `candidate_<model>.joblib` và `candidate_<model>.json`.
3. **`check_model` (`@task.branch`)**: Chọn mô hình có `accuracy` cao nhất (`best_candidate.json`) và so sánh với `prod_metrics.json`:
   - Nếu `best["accuracy"] >= prod_acc` → đi nhánh **`promote_and_reload`**.
   - Nếu kém hơn → đi nhánh **`skip_deploy`** (giữ nguyên mô hình production cũ).
4. **`promote_and_reload`**: Copy sang `prod_model.joblib` & `prod_metrics.json`, sau đó gửi `POST http://ml-api:8000/reload` để FastAPI nạp lại mô hình mới vào RAM mà không cần restart container.

### Cơ chế chia sẻ dữ liệu (Docker Shared Volume)
Tất cả container (`ml-api`, `airflow-scheduler`, và các container ML tạm thời của `DockerOperator`) đều mount cùng một Docker Named Volume **`airflow_mlops_shared`** vào đường dẫn **`/opt/shared`** và nằm chung mạng **`airflow_mlops_net`**:

```text
data_processing.py  ──► /opt/shared/raw.csv, processed.csv
training.py (x3)    ──► /opt/shared/candidate_<model>.joblib, candidate_<model>.json
check_model()       ──► /opt/shared/best_candidate.json
promote_and_reload()──► /opt/shared/prod_model.joblib, prod_metrics.json ──► api.py
```

---

## 2. Cấu trúc thư mục

```text
airflow-mlops/
├── airflow/
│   └── Dockerfile                 # Image Airflow 3.4.0 cài thêm providers-docker & providers-http
├── dags/
│   └── penguins_ml_pipeline.py    # Định nghĩa DAG Airflow 3 (Task SDK + DockerOperator)
├── data/
│   └── raw/penguins.csv           # Dữ liệu mẫu tham khảo
├── ml/
│   ├── Dockerfile                 # Image ML chung cho cả ml-api và các container worker
│   ├── requirements.txt           # Thư viện Python cho ML & FastAPI
│   ├── data_processing.py         # Bước 1: tạo batch mới & làm sạch dữ liệu
│   ├── training.py                # Bước 2: train model theo tham số CLI (sys.argv[1])
│   └── api.py                     # FastAPI serving (/health, /reload, /predict)
├── docker-compose.yaml            # Cụm Airflow 3.4.0 (LocalExecutor: postgres, init, apiserver, dag-processor, scheduler)
├── docker-compose.mlops.yaml      # Service ml-api + khai báo shared volume & network
└── .env                           # Biến môi trường (UID/GID, Postgres, JWT secret, đường dẫn chung)
```

---

## 3. Chuẩn bị môi trường (`.env`)

Yêu cầu: **Docker** và **Docker Compose v2**.

Kiểm tra UID người dùng và GID của Docker socket trên máy Linux:
```bash
id -u
stat -c '%g' /var/run/docker.sock
```

Tạo hoặc cập nhật file `.env` ở thư mục gốc theo giá trị trên máy của bạn:
```env
AIRFLOW_UID=1000
DOCKER_GID=116

POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow

AIRFLOW__API_AUTH__JWT_SECRET=airflow-mlops-dev-jwt-secret-key

SHARED_DATA_DIR=/opt/shared
ML_API_URL=http://ml-api:8000
```

---

## 4. Khởi chạy hệ thống

Đứng tại thư mục gốc `airflow-mlops/`:

### Cách 1: Chạy từng phần (dễ kiểm tra từng tầng)
```bash
# Bước 1: Build image ML (airflow-mlops-ml:latest) và bật FastAPI
docker compose -f docker-compose.mlops.yaml up -d --build

# Bước 2: Build image Airflow 3.4.0 và bật cụm Airflow
docker compose up -d --build
```

### Cách 2: Chạy gộp cả hệ thống trong 1 lệnh
```bash
docker compose -f docker-compose.yaml -f docker-compose.mlops.yaml up -d --build
```

---

## 5. Kiểm tra & Sử dụng

- **Airflow 3 UI:** `http://localhost:8080` *(đang bật `SimpleAuthManager` với `ALL_ADMINS=True` cho môi trường dev local — vào thẳng giao diện quản trị).*
- **FastAPI Swagger UI:** `http://localhost:8000/docs`

### Kiểm tra trạng thái API
```bash
curl http://localhost:8000/health
```

### Gọi dự đoán sau khi DAG chạy xong ít nhất 1 lượt
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750
  }'
```

### Theo dõi quá trình cập nhật file trong shared volume (`/opt/shared`)
```bash
# Xem live danh sách file và metrics thay đổi mỗi 2 giây
watch -n 2 "docker exec ml-api sh -c 'ls -lh /opt/shared; echo ===PROD===; cat /opt/shared/prod_metrics.json 2>/dev/null; echo ===BEST===; cat /opt/shared/best_candidate.json 2>/dev/null'"
```

---

## 6. Quy tắc áp dụng khi chỉnh sửa code / cấu hình

| File bạn vừa sửa | Lệnh cần chạy lại |
|---|---|
| `dags/penguins_ml_pipeline.py` | Không cần chạy lại (`airflow-dag-processor` tự quét sau ~30 giây) |
| `docker-compose.yaml`, `docker-compose.mlops.yaml`, `.env` | `docker compose -f docker-compose.yaml -f docker-compose.mlops.yaml up -d` |
| `ml/*.py`, `ml/requirements.txt`, `ml/Dockerfile` | `docker compose -f docker-compose.mlops.yaml up -d --build` |
| `airflow/Dockerfile` | `docker compose up -d --build` |

---

## 7. Dừng & Dọn dẹp

```bash
# Dừng toàn bộ container (giữ lại dữ liệu Postgres và model trong shared volume)
docker compose -f docker-compose.yaml -f docker-compose.mlops.yaml down

# Dừng và xóa sạch cả volume dữ liệu để chạy lại từ đầu
docker compose -f docker-compose.yaml -f docker-compose.mlops.yaml down -v
```
