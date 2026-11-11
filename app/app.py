import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.data import load_data, RISK_ORDER, EXPECTED_CATEGORIES
from src.preprocessing import prepare_data

MODEL_PATH = ROOT / "models" / "best_pipeline.joblib"
DATA_PATHS = [ROOT / "data" / "raw" / "student_mental_health.csv",
              ROOT / "student_mental_health.csv"]

st.set_page_config(page_title="Student mental health risk", layout="wide")
st.title("Student mental health risk")
st.caption("Educational project: the prediction is not a medical diagnosis.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_splits():
    path = next(p for p in DATA_PATHS if p.exists())
    return prepare_data(load_data(path))


if not MODEL_PATH.exists():
    st.error("Model not found. Run notebook 04 or the Airflow DAG to create models/best_pipeline.joblib.")
    st.stop()
if not any(p.exists() for p in DATA_PATHS):
    st.error("Dataset student_mental_health.csv not found.")
    st.stop()

model = load_model()
X_train, X_test, y_train, y_test = load_splits()

tab_perf, tab_one, tab_csv = st.tabs(["Model performance", "Predict one student", "Predict from a CSV"])

# 1. performance on the test set (same split and metrics as notebook 04)
with tab_perf:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="macro"),
        "Recall": recall_score(y_test, pred, average="macro"),
        "F1": f1_score(y_test, pred, average="macro"),
        "ROC-AUC": roc_auc_score(y_test, proba, multi_class="ovr", average="macro"),
    }
    for col, (name, value) in zip(st.columns(5), metrics.items()):
        col.metric(name, f"{value:.3f}")
    st.caption(f"Macro averages on the test set ({len(X_test)} students).")

    left, right = st.columns(2)
    left.subheader("Confusion matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=RISK_ORDER).plot(ax=ax, colorbar=False)
    left.pyplot(fig)

    report = pd.DataFrame(classification_report(y_test, pred, target_names=RISK_ORDER, output_dict=True)).T.loc[RISK_ORDER]
    right.subheader("F1 score per risk level")
    right.bar_chart(report["f1-score"])
    right.caption(f"Weakest class: {report['f1-score'].idxmin()}")

# 2. one student, typed in a form
with tab_one:
    values = {}
    columns = st.columns(3)
    for i, col in enumerate(X_train.columns):
        box, series = columns[i % 3], X_train[col]
        if pd.api.types.is_numeric_dtype(series):
            cast = int if pd.api.types.is_integer_dtype(series) else float
            values[col] = box.number_input(col, min_value=cast(series.min()), max_value=cast(series.max()),
                                           value=cast(series.median()))
        else:
            options = EXPECTED_CATEGORIES[col]
            values[col] = box.selectbox(col, options, index=options.index(series.mode()[0]))

    if st.button("Predict"):
        if values["Screen_Time_Hours"] < values["Social_Media_Hours"]:
            st.error("Screen time must be at least the social media time.")
        else:
            row = pd.DataFrame([values])
            st.subheader(f"Predicted risk: {RISK_ORDER[model.predict(row)[0]]}")
            st.bar_chart(pd.Series(model.predict_proba(row)[0], index=RISK_ORDER))

# 3. many students, from a CSV file
with tab_csv:
    st.write("Required columns: " + ", ".join(X_train.columns))
    st.download_button("Download a sample file", X_test.head(20).to_csv(index=False), "sample.csv")
    file = st.file_uploader("CSV file", type="csv")
    if file:
        new = pd.read_csv(file)
        missing = [c for c in X_train.columns if c not in new.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            result = new.assign(Predicted_risk=[RISK_ORDER[i] for i in model.predict(new[X_train.columns])])
            st.dataframe(result)
            st.bar_chart(result["Predicted_risk"].value_counts().reindex(RISK_ORDER, fill_value=0))
            st.download_button("Download predictions", result.to_csv(index=False), "predictions.csv")