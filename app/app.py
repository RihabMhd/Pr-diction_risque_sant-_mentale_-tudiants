from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RISK_ORDER = ["Low", "Medium", "High"]

model = joblib.load(ROOT / "models" / "best_pipeline.joblib")
data = pd.read_csv(ROOT / "data" / "raw" / "student_mental_health.csv")
X = data.drop(columns=["Student_ID", "Mental_Health_Risk"])

st.title("Understanding Student Mental Health")
st.caption("A simple project to explore students' mental health and predict their risk level")

values = {}
columns = st.columns(3)
for i, col in enumerate(X.columns):
    box, series = columns[i % 3], X[col]
    if pd.api.types.is_numeric_dtype(series):
        cast = int if pd.api.types.is_integer_dtype(series) else float
        values[col] = box.number_input(col, cast(series.min()), cast(series.max()), cast(series.median()))
    else:
        values[col] = box.selectbox(col, series.dropna().unique())

if st.button("Predict"):
    if values["Screen_Time_Hours"] < values["Social_Media_Hours"]:
        st.error("Your total screen time can't be less than your social media time")
    else:
        row = pd.DataFrame([values])
        st.subheader(f"Predicted : {RISK_ORDER[model.predict(row)[0]]}")
        st.bar_chart(pd.Series(model.predict_proba(row)[0], index=RISK_ORDER))