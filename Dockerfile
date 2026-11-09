FROM apache/airflow:2.9.2-python3.10

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir \
    "apache-airflow==2.9.2" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.2/constraints-3.10.txt" \
    -r /requirements.txt