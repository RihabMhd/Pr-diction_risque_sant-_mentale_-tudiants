import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

BASE = Path(os.getenv("PROJECT_DIR", "/opt/airflow"))
sys.path.insert(0, str(BASE))

RAW_PATH = BASE / "data" / "raw" / "student_mental_health.csv"
CLEAN_DIR = BASE / "data" / "clean"
MODEL_PATH = BASE / "models" / "best_pipeline.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"

log = logging.getLogger(__name__)


def notify_failure(context):
    ti = context["task_instance"]
    log.error("FAILED task=%s run=%s try=%s error=%s",
              ti.task_id, context.get("run_id"), ti.try_number, context.get("exception"))


@dag(
    dag_id="student_mental_health_pipeline",
    schedule="@weekly",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
        "execution_timeout": timedelta(minutes=30),
        "on_failure_callback": notify_failure,
    },
    tags=["ml", "mental-health"],
)
def student_mental_health_pipeline():

    @task
    def audit():
        from src.data import load_data, get_column_types
        from src.eda import audit_data

        if not RAW_PATH.exists():
            raise AirflowFailException(f"Raw data not found: {RAW_PATH}")

        df = load_data(RAW_PATH)
        _, continuous_cols, _ = get_column_types(df)
        report = audit_data(df, continuous_cols)

        CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        (CLEAN_DIR / "audit.json").write_text(json.dumps(report, indent=2))
        log.info("Audit: %s", report)
        return report

    @task
    def preprocess():
        from src.data import load_data, TARGET
        from src.preprocessing import prepare_data

        X_train, X_test, y_train, y_test = prepare_data(load_data(RAW_PATH))

        CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        X_train.assign(**{TARGET: y_train}).to_csv(CLEAN_DIR / "train.csv", index=False)
        X_test.assign(**{TARGET: y_test}).to_csv(CLEAN_DIR / "test.csv", index=False)
        log.info("train=%s test=%s", X_train.shape, X_test.shape)
        return {"train_rows": len(X_train), "test_rows": len(X_test)}

    @task
    def train():
        import pandas as pd
        from src.data import TARGET
        from src.tuning import tune_models

        train_df = pd.read_csv(CLEAN_DIR / "train.csv")
        test_df = pd.read_csv(CLEAN_DIR / "test.csv")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        result = tune_models(
            train_df.drop(columns=TARGET), test_df.drop(columns=TARGET),
            train_df[TARGET], test_df[TARGET], MODEL_PATH,
        )
        log.info("Best model: %s | CV F1: %s | test: %s",
                 result["best_model"], result["cv_f1"], result["test"])
        return result

    @task
    def evaluate(result):
        import joblib
        import pandas as pd
        from sklearn.metrics import f1_score
        from src.data import TARGET

        test_df = pd.read_csv(CLEAN_DIR / "test.csv")
        pipeline = joblib.load(MODEL_PATH)
        f1 = f1_score(test_df[TARGET], pipeline.predict(test_df.drop(columns=TARGET)), average="macro")

        if abs(f1 - result["test"]["f1"]) > 1e-6:
            raise AirflowFailException(
                f"Saved pipeline F1 {f1:.4f} != training run F1 {result['test']['f1']:.4f}")

        METRICS_PATH.write_text(json.dumps(
            {**result, "trained_at": datetime.now().isoformat(timespec="seconds")}, indent=2))
        log.info("Saved %s and %s (F1=%.3f)", MODEL_PATH, METRICS_PATH, f1)

    audit_out, preprocess_out, train_out = audit(), preprocess(), train()
    audit_out >> preprocess_out >> train_out
    evaluate(train_out)


student_mental_health_pipeline()