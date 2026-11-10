import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

log = logging.getLogger(__name__)

BASE = Path(os.getenv("PROJECT_DIR", "/opt/airflow"))
RAW_PATH = BASE / "data/raw/student_mental_health.csv"
SPLIT_PATH = BASE / "data/clean/split.joblib"
MODEL_PATH = BASE / "models/best_pipeline.joblib"
REPORTS = BASE / "reports"


def _write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def notify_failure(context):
    ti = context["task_instance"]
    log.error(
        "FAILED dag=%s task=%s run=%s try=%s error=%s",
        ti.dag_id, ti.task_id, context["run_id"], ti.try_number, context.get("exception"),
    )


@dag(
    dag_id="mental_health_risk_pipeline",
    schedule="0 2 * * 1",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=1),
    default_args={
        "owner": "rihab",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
        "execution_timeout": timedelta(minutes=30),
        "on_failure_callback": notify_failure,
    },
    tags=["ml", "mental-health"],
)
def mental_health_pipeline():

    @task
    def audit():
        from src.data import load_data, get_column_types
        from src.eda import audit_data

        df = load_data(RAW_PATH)
        if df.empty:
            raise ValueError(f"Dataset vide : {RAW_PATH}")
        numeric, _ = get_column_types(df)
        report = audit_data(df, numeric)
        _write_json(report, REPORTS / "audit.json")
        return {"shape": report["shape"], "imbalance_ratio": report["imbalance_ratio"]}

    @task
    def preprocess():
        import joblib
        from src.data import load_data
        from src.preprocessing import prepare_data

        split = prepare_data(load_data(RAW_PATH))  
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(split, SPLIT_PATH)
        return {"n_train": len(split[0]), "n_test": len(split[1])}

    @task
    def train():
        import joblib
        from src.models import train_models

        _, results = train_models(*joblib.load(SPLIT_PATH))
        _write_json(results, REPORTS / "models_metrics.json")
        return {n: round(float(m["f1"]), 4) for n, m in results.items()}

    @task
    def tune_evaluate():
        import joblib
        from src.tuning import tune_models

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        final = tune_models(*joblib.load(SPLIT_PATH), model_path=str(MODEL_PATH))
        _write_json(final, REPORTS / "final_metrics.json")
        return {"best_model": final["best_model"], "test_f1": float(final["test"]["f1"])}

    audit() >> preprocess() >> train() >> tune_evaluate()


mental_health_pipeline()