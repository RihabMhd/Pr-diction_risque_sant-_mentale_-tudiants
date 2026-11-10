import pandas as pd

DATA_PATH = "../data/raw/student_mental_health.csv"   
TARGET = "Mental_Health_Risk"
RISK_ORDER = ["Low", "Medium", "High"]

EXPECTED_CATEGORIES = {
    "Gender": ["Male", "Female", "Non-binary"],
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Part_Time_Job": ["Yes", "No"],
    "Diet_Quality": ["Poor", "Average", "Good"],
    "Previous_Mental_Health_History": ["Yes", "No"],
    "Mental_Health_Risk": ["Low", "Medium", "High"],
}


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def get_column_types(df):
    """return numeric columns, categorical columns"""
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != "Student_ID"
    ]
    categorical_cols = [
        c for c in df.select_dtypes(include="object").columns
        if c not in ["Student_ID", TARGET]
    ]
    return numeric_cols, categorical_cols