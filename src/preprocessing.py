import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from .data import TARGET, RISK_ORDER, EXPECTED_CATEGORIES

RANDOM_STATE = 42
RISK_CODES = {risk: i for i, risk in enumerate(RISK_ORDER)} 

ORDINAL_COLS = {
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Diet_Quality": ["Poor", "Average", "Good"],
}

def clean_data(df):
    """row-level cleaning. nothing here is learned from the data,
    so doing it before the split does not cause data leakage"""
    df = df.copy()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    df = df.drop_duplicates()

    for col, valid_values in EXPECTED_CATEGORIES.items():
        is_invalid = df[col].notna() & ~df[col].isin(valid_values)
        df.loc[is_invalid, col] = np.nan
        
    df = df.dropna(subset=[TARGET])

    return df.reset_index(drop=True)


def split_data(df):
    """80% train / 20% test, same class proportions in both"""
    X = df.drop(columns=[TARGET, "Student_ID"]) 
    y = df[TARGET].map(RISK_CODES)
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )


class IQRClipper(BaseEstimator, TransformerMixin):
    """replace values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] by the nearest bound
    the bounds are learned on the training data only (fit)"""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - 1.5 * iqr).values
        self.upper_ = (q3 + 1.5 * iqr).values
        return self

    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features)


def build_preprocessor(X_train):
    numeric_cols = X_train.select_dtypes(include="number").columns.to_list()
    ordinal_cols = list(ORDINAL_COLS.keys())
    nominal_cols = [
        c for c in X_train.select_dtypes(include="object").columns
        if c not in ordinal_cols
    ]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outliers", IQRClipper()),
        ("scaler", StandardScaler()),
    ])
    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=list(ORDINAL_COLS.values()))),
        ("scaler", StandardScaler()),
    ])
    nominal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("ordinal", ordinal_pipe, ordinal_cols),
        ("nominal", nominal_pipe, nominal_cols),
    ])


def build_pipeline(model, X_train, use_smote=False):
    steps = [("preprocessing", build_preprocessor(X_train))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return ImbPipeline(steps)