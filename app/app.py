import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

BASE = Path(os.getenv("PROJECT_DIR", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(BASE))
from src.data import EXPECTED_CATEGORIES, RISK_ORDER, TARGET  # noqa: E402

MODEL_PATH = BASE / "models" / "best_pipeline.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"
TRAIN_PATH = BASE / "data" / "clean" / "train.csv"
TEST_PATH = BASE / "data" / "clean" / "test.csv"

st.set_page_config(page_title="Student Mental Health Risk", layout="wide")


@st.cache_resource
def load_model(mtime):
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv(path, mtime):
    return pd.read_csv(path)


@st.cache_data
def load_metrics(mtime):
    return json.loads(METRICS_PATH.read_text())


def is_int_like(s):
    return bool((s.dropna() % 1 == 0).all())


st.title("Student mental health risk")
st.caption("Educational project: risk screening model, not a diagnostic tool.")

absent = [p.name for p in (MODEL_PATH, METRICS_PATH, TRAIN_PATH, TEST_PATH) if not p.exists()]
if absent:
    st.error(f"Missing: {', '.join(absent)}. Run the Airflow DAG first.")
    st.stop()

model = load_model(MODEL_PATH.stat().st_mtime)
metrics = load_metrics(METRICS_PATH.stat().st_mtime)
train = load_csv(TRAIN_PATH, TRAIN_PATH.stat().st_mtime)
test = load_csv(TEST_PATH, TEST_PATH.stat().st_mtime)

features = [c for c in train.columns if c != TARGET]
X_test, y_test = test[features], test[TARGET]

tab_perf, tab_predict = st.tabs(["Model performance", "Test the model"])

with tab_perf:
    st.subheader(f"Model: {metrics['best_model']}")
    st.caption(f"Trained at {metrics.get('trained_at', 'n/a')} | evaluated on {len(test)} held-out students")

    t = metrics["test"]
    cols = st.columns(5)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(key.replace("_", " ").title(), f"{t[key]:.3f}")
    st.caption("Precision, recall and F1 are macro averages; ROC-AUC is one-vs-rest.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Cross-validated F1 (train) per tuned model**")
        st.bar_chart(pd.Series(metrics["cv_f1"], name="CV F1"))
        with st.expander("Best hyperparameters"):
            st.json(metrics["best_params"])
    with right:
        st.markdown("**Per-class report (test)**")
        pred = model.predict(X_test)
        report = classification_report(y_test, pred, target_names=RISK_ORDER, output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3))

    st.markdown("**Confusion matrix (test): counts and row-normalized (recall)**")
    cm = np.array(t["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ConfusionMatrixDisplay(cm, display_labels=RISK_ORDER).plot(ax=axes[0], colorbar=False)
    ConfusionMatrixDisplay(cm / cm.sum(axis=1, keepdims=True), display_labels=RISK_ORDER).plot(
        ax=axes[1], colorbar=False, values_format=".2f")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**ROC curves (test, one-vs-rest)**")
    proba = model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, risk in enumerate(RISK_ORDER):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        ax.plot(fpr, tpr, label=f"{risk} (AUC={roc_auc_score(y_bin[:, i], proba[:, i]):.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    st.pyplot(fig)

with tab_predict:
    mode = st.radio("Input", ["One student", "CSV file"], horizontal=True)

    if mode == "One student":
        with st.form("single"):
            cols = st.columns(3)
            values = {}
            for i, col in enumerate(features):
                s = train[col]
                with cols[i % 3]:
                    if col in EXPECTED_CATEGORIES or s.dtype == object:
                        options = EXPECTED_CATEGORIES.get(col) or sorted(s.dropna().unique())
                        values[col] = st.selectbox(col, options)
                    elif is_int_like(s):
                        values[col] = st.number_input(col, int(s.min()), int(s.max()), int(s.median()), step=1)
                    else:
                        values[col] = st.number_input(col, float(s.min()), float(s.max()), float(s.median()), step=0.1)
            submitted = st.form_submit_button("Predict")

        if submitted:
            row = pd.DataFrame([values])[features]
            proba = model.predict_proba(row)[0]
            st.metric("Predicted risk", RISK_ORDER[int(model.predict(row)[0])])
            st.bar_chart(pd.Series(proba, index=RISK_ORDER, name="Probability"))

    else:
        st.download_button("Download CSV template", X_test.head(5).to_csv(index=False), "template.csv")
        file = st.file_uploader("CSV with the same columns as the template", type="csv")
        if file:
            new = pd.read_csv(file)
            missing = [c for c in features if c not in new.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                try:
                    proba = model.predict_proba(new[features])
                except ValueError as e:
                    st.error(f"Invalid values: {e}")
                else:
                    out = new.copy()
                    out["Predicted_Risk"] = [RISK_ORDER[int(model.classes_[i])] for i in proba.argmax(axis=1)]
                    for i, risk in enumerate(RISK_ORDER):
                        out[f"P({risk})"] = proba[:, i].round(3)
                    st.dataframe(out)
                    st.bar_chart(out["Predicted_Risk"].value_counts().reindex(RISK_ORDER, fill_value=0))
                    st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv")