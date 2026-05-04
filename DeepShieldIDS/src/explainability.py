
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier


def permutation_feature_importance(X_train, y_train, X_test, y_test, output_csv, top_k=30):
    clf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    result = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    df = pd.DataFrame({
        "feature": X_train.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).head(top_k)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def shap_summary_optional(model, X_sample, output_png):
    try:
        import shap
        import matplotlib.pyplot as plt
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        shap.summary_plot(shap_values, X_sample, show=False)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_png, dpi=300)
        plt.close()
        return True
    except Exception:
        return False
