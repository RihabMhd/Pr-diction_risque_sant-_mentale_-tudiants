import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow.decorators import dag, task

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "/opt/airflow"))
RAW_CSV = PROJECT_DIR / "student_mental_health.csv"
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
REPORTS_DIR = PROJECT_DIR / "reports"
MODEL_PATH = PROJECT_DIR / "models" / "best_pipeline.joblib"


def log_failure(context):
    logging.error("Task %s failed: %s", context["task_instance"].task_id, context.get("exception"))


def save_json(content, name):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / name).write_text(json.dumps(content, indent=2))


def load_splits():
    X_train = pd.read_csv(CLEAN_DIR / "X_train.csv")
    X_test = pd.read_csv(CLEAN_DIR / "X_test.csv")
    y_train = pd.read_csv(CLEAN_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(CLEAN_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


@task
def audit():
    """data-quality audit of the raw CSV -> reports/audit.json"""
    from src.data import load_data, get_column_types
    from src.eda import audit_data

    df = load_data(RAW_CSV)
    numeric_cols, _ = get_column_types(df)
    save_json(audit_data(df, numeric_cols), "audit.json")


@task
def preprocess():
    """clean the data and split 80/20 -> data/clean/*.csv"""
    from src.data import load_data
    from src.preprocessing import prepare_data

    X_train, X_test, y_train, y_test = prepare_data(load_data(RAW_CSV))
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}.items():
        part.to_csv(CLEAN_DIR / f"{name}.csv", index=False)


@task
def train():
    """train the baseline models -> reports/train_metrics.json"""
    from src.models import train_models

    _, results = train_models(*load_splits())
    save_json(results, "train_metrics.json")


@task
def tune_evaluate():
    """tune, pick the best model by CV, evaluate on test -> models/best_pipeline.joblib + reports/final_metrics.json"""
    from src.tuning import tune_models

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = tune_models(*load_splits(), str(MODEL_PATH))
    save_json(results, "final_metrics.json")


@dag(
    dag_id="mental_health_risk_pipeline",
    schedule="@weekly",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2), "on_failure_callback": log_failure},
    tags=["mental-health"],
)
def mental_health_risk_pipeline():
    audit() >> preprocess() >> train() >> tune_evaluate()


mental_health_risk_pipeline()