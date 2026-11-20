FROM apache/airflow:3.4.0

USER airflow

RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.13.txt" \
    apache-airflow-providers-docker \
    apache-airflow-providers-http
