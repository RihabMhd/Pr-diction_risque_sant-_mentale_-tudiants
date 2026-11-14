import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve,
)
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.data import load_data
from src.preprocessing import prepare_data

RISK_ORDER = ["Low", "Medium", "High"]

st.set_page_config(page_title="Student Mental Health", layout="wide")


@st.cache_resource
def get_model():
    return joblib.load(ROOT / "models" / "best_pipeline.joblib")


@st.cache_data
def get_data():
    data = load_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    return data, X_test, y_test


model = get_model()
data, X_test, y_test = get_data()
X = data.drop(columns=["Student_ID", "Mental_Health_Risk"])

st.title("Student mental health risk")

tab_perf, tab_predict = st.tabs(["Model performance", "Test the model"])

with tab_perf:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    st.write("Model:", type(model.named_steps["model"]).__name__)
    st.write("Evaluated on", len(X_test), "students not used for training.")

    cols = st.columns(5)
    cols[0].metric("Accuracy", round(accuracy_score(y_test, pred), 3))
    cols[1].metric("Precision", round(precision_score(y_test, pred, average="macro"), 3))
    cols[2].metric("Recall", round(recall_score(y_test, pred, average="macro"), 3))
    cols[3].metric("F1", round(f1_score(y_test, pred, average="macro"), 3))
    cols[4].metric("ROC AUC", round(roc_auc_score(y_test, proba, multi_class="ovr"), 3))

    left, right = st.columns(2)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=RISK_ORDER).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion matrix")
    left.pyplot(fig)

    fig, ax = plt.subplots()
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    for i, risk in enumerate(RISK_ORDER):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        ax.plot(fpr, tpr, label=risk)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend()
    right.pyplot(fig)

    st.subheader("Results by class")
    report = classification_report(y_test, pred, target_names=RISK_ORDER, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3))

with tab_predict:
    st.subheader("One student")

    values = {}
    columns = st.columns(3)
    for i, col in enumerate(X.columns):
        box, series = columns[i % 3], X[col]
        label = col.replace("_", " ")
        if pd.api.types.is_numeric_dtype(series):
            cast = int if pd.api.types.is_integer_dtype(series) else float
            values[col] = box.number_input(label, cast(series.min()), cast(series.max()), cast(series.median()))
        else:
            values[col] = box.selectbox(label, series.dropna().unique())

    if st.button("Predict"):
        if values["Screen_Time_Hours"] < values["Social_Media_Hours"]:
            st.error("Screen time cannot be lower than social media time.")
        else:
            row = pd.DataFrame([values])
            st.write("Predicted risk:", RISK_ORDER[model.predict(row)[0]])
            st.bar_chart(pd.Series(model.predict_proba(row)[0], index=RISK_ORDER))

    st.subheader("Several students (CSV)")

    file = st.file_uploader("CSV with the same columns as the training data", type="csv")
    if file:
        new = pd.read_csv(file)
        missing = [c for c in X.columns if c not in new.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            result = new.copy()
            result["Predicted_risk"] = [RISK_ORDER[i] for i in model.predict(new[X.columns])]
            st.dataframe(result)
            st.download_button("Download predictions", result.to_csv(index=False), "predictions.csv")