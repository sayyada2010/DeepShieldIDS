from __future__ import annotations
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_sklearn_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_score)
        else:
            auc = None
    except Exception:
        auc = None
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc": auc,
    }


def run_baselines(X_train, y_train, X_test, y_test, output_path):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "LinearSVM": CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=5000)),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=42, n_estimators=150, max_depth=6)
    except Exception:
        pass
    rows = []
    for name, model in models.items():
        scores = evaluate_sklearn_model(model, X_train, y_train, X_test, y_test)
        scores["model"] = name
        rows.append(scores)
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
