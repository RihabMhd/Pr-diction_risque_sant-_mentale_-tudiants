from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from .data import get_column_types
from .preprocessing import RANDOM_STATE

ORDINAL_COLS = {
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Diet_Quality": ["Poor", "Average", "Good"],
}


def train_models(X_train, X_test, y_train, y_test):
    """train 3 models on the same train set. Returns (fitted pipelines, JSON-serializable test metrics)"""
    numeric, _, categorical = get_column_types(X_train)
    ordinal = list(ORDINAL_COLS)
    nominal = [c for c in categorical if c not in ordinal]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", RobustScaler())]), numeric),
        ("ord", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OrdinalEncoder(categories=list(ORDINAL_COLS.values())))]), ordinal),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]), nominal),
    ])

    counts = y_train.value_counts()
    use_smote = counts.max() / counts.min() > 1.5

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    pipelines, results = {}, {}
    for name, model in models.items():
        steps = [("preprocessing", preprocessor)]
        if use_smote:
            steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        steps.append(("model", model))

        pipe = ImbPipeline(steps).fit(X_train, y_train)
        pred, proba = pipe.predict(X_test), pipe.predict_proba(X_test)

        pipelines[name] = pipe
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, average="macro"),
            "recall": recall_score(y_test, pred, average="macro"),
            "f1": f1_score(y_test, pred, average="macro"),
            "roc_auc": roc_auc_score(y_test, proba, multi_class="ovr", average="macro"),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }

    return pipelines, results