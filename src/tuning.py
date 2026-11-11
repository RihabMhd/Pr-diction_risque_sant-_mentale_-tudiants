import joblib
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .models import train_models
from .preprocessing import RANDOM_STATE

SCORING = "f1_macro"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

LOGREG_GRID = {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]}
SVC_PARAMS = {
    "model__estimator__C": [1, 3, 10, 30, 100, 300],
    "model__estimator__gamma": [0.0001, 0.0003, 0.001, 0.003, 0.01, "scale"],
}


def tune_models(X_train, X_test, y_train, y_test, model_path):
    """tune LogisticRegression (grid) and SVC (grid) with CV on train,
    keep the best by CV F1, evaluate it once on test, save the full pipeline to model_path"""
    base, _ = train_models(X_train, X_test, y_train, y_test)

    searches = {
        "LogisticRegression": GridSearchCV(
            base["LogisticRegression"], LOGREG_GRID, scoring=SCORING, cv=CV, n_jobs=-1),
        "SVC": GridSearchCV(
            base["SVC"], SVC_PARAMS, scoring=SCORING, cv=CV, n_jobs=-1),
    }
    for search in searches.values():
        search.fit(X_train, y_train)

    best_name = max(searches, key=lambda n: searches[n].best_score_)
    best = searches[best_name].best_estimator_
    pred, proba = best.predict(X_test), best.predict_proba(X_test)
    joblib.dump(best, model_path)

    return {
        "best_model": best_name,
        "cv_f1": {n: float(s.best_score_) for n, s in searches.items()},
        "best_params": {n: {k: (v.item() if hasattr(v, "item") else v)
                            for k, v in s.best_params_.items()} for n, s in searches.items()},
        "test": {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, average="macro"),
            "recall": recall_score(y_test, pred, average="macro"),
            "f1": f1_score(y_test, pred, average="macro"),
            "roc_auc": roc_auc_score(y_test, proba, multi_class="ovr", average="macro"),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        },
    }