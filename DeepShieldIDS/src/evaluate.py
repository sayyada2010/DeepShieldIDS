
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
from .model import to_sequence
from .config import FIGURE_DIR, METRICS_DIR


def predict_labels(model, X, binary=True):
    probs = model.predict(to_sequence(X), verbose=0)
    if binary:
        prob1 = probs.ravel()
        pred = (prob1 >= 0.5).astype(int)
        return pred, prob1
    pred = probs.argmax(axis=1)
    return pred, probs


def compute_metrics(y_true, y_pred, y_score=None, binary=True):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred)
    if binary and cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        out["fpr"] = float(fp / (fp + tn + 1e-12))
    else:
        out["fpr"] = None
    try:
        if binary:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        else:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score, multi_class="ovr"))
    except Exception:
        out["roc_auc"] = None
    return out


def plot_confusion_matrix(y_true, y_pred, run_name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title(f"Confusion Matrix - {run_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{run_name}_confusion_matrix.png", dpi=300)
    plt.close(fig)


def plot_roc(y_true, y_score, run_name):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(f"ROC Curve - {run_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{run_name}_roc_curve.png", dpi=300)
    plt.close(fig)


def evaluate_model(model, X_test, y_test, binary=True, run_name="hybrididsnet"):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    y_pred, y_score = predict_labels(model, X_test, binary=binary)
    metrics = compute_metrics(y_test, y_pred, y_score, binary=binary)
    pd.DataFrame([metrics]).to_csv(METRICS_DIR / f"{run_name}_metrics.csv", index=False)
    with open(METRICS_DIR / f"{run_name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))
    plot_confusion_matrix(y_test, y_pred, run_name)
    if binary:
        plot_roc(y_test, y_score, run_name)
    return metrics
