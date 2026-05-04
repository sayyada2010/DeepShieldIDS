
from __future__ import annotations
import optuna
from .train import train_hybrididsnet
from .evaluate import evaluate_model


def tune_optuna(X_train, y_train, X_val, y_val, binary=True, n_trials=10):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "cnn_filters": trial.suggest_categorical("cnn_filters", [32, 64, 128]),
            "lstm_units": trial.suggest_categorical("lstm_units", [64, 128, 256]),
            "dropout": trial.suggest_categorical("dropout", [0.2, 0.3, 0.5]),
            "epochs": 15,
            "patience": 4,
        }
        model, _ = train_hybrididsnet(X_train, y_train, X_val, y_val, binary=binary, params=params, run_name=f"optuna_trial_{trial.number}")
        metrics = evaluate_model(model, X_val, y_val, binary=binary, run_name=f"optuna_trial_{trial.number}")
        return metrics["f1_macro"]
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value
