import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

BASE = Path("/opt/airflow")
sys.path.insert(0, str(BASE))

RAW_FILE = BASE / "data/raw/student_mental_health.csv"
CLEAN_DIR = BASE / "data/clean"
MODEL_FILE = BASE / "models/best_pipeline.joblib"
METRICS_FILE = BASE / "models/metrics.json"

log = logging.getLogger(__name__)


@dag(
    dag_id="student_mental_health_pipeline",
    schedule="@weekly",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["ml", "mental-health"],
)
def student_mental_health_pipeline():

    @task
    def audit():
        from src.data import load_data, get_column_types
        from src.eda import audit_data

        if not RAW_FILE.exists():
            raise AirflowFailException("Dataset not found")

        df = load_data(RAW_FILE)
        _, numeric_columns, _ = get_column_types(df)
        report = audit_data(df, numeric_columns)

        CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        (CLEAN_DIR / "audit.json").write_text(json.dumps(report, indent=2))
        log.info("Data audit completed")

    @task
    def preprocess():
        from src.data import load_data, TARGET
        from src.preprocessing import prepare_data

        df = load_data(RAW_FILE)
        X_train, X_test, y_train, y_test = prepare_data(df)

        CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        X_train.assign(**{TARGET: y_train}).to_csv(CLEAN_DIR / "train.csv", index=False)
        X_test.assign(**{TARGET: y_test}).to_csv(CLEAN_DIR / "test.csv", index=False)

    @task
    def train():
        import pandas as pd
        from src.data import TARGET
        from src.tuning import tune_models

        train_df = pd.read_csv(CLEAN_DIR / "train.csv")
        test_df = pd.read_csv(CLEAN_DIR / "test.csv")
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

        return tune_models(
            train_df.drop(columns=TARGET),
            test_df.drop(columns=TARGET),
            train_df[TARGET],
            test_df[TARGET],
            MODEL_FILE,
        )

    @task
    def evaluate(results):
        import joblib
        import pandas as pd
        from sklearn.metrics import f1_score
        from src.data import TARGET

        test_df = pd.read_csv(CLEAN_DIR / "test.csv")
        X_test = test_df.drop(columns=TARGET)
        y_test = test_df[TARGET]

        model = joblib.load(MODEL_FILE)
        score = f1_score(y_test, model.predict(X_test), average="macro")

        if abs(score - results["test"]["f1"]) > 1e-6:
            raise AirflowFailException("Saved model score does not match training score")

        results["trained_at"] = datetime.now().isoformat(timespec="seconds")
        METRICS_FILE.write_text(json.dumps(results, indent=2))
        log.info("Evaluation completed. F1-score: %.3f", score)

    audit_task = audit()
    preprocess_task = preprocess()
    train_task = train()

    audit_task >> preprocess_task >> train_task
    evaluate(train_task)


student_mental_health_pipeline()
