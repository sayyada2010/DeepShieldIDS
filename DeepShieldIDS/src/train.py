
from __future__ import annotations
from pathlib import Path
import pandas as pd
import tensorflow as tf
from .model import build_hybrididsnet, to_sequence
from .config import DEFAULT_MODEL_PARAMS, MODEL_DIR, METRICS_DIR


def train_hybrididsnet(X_train, y_train, X_val, y_val, binary=True, params=None, run_name="hybrididsnet"):
    params = {**DEFAULT_MODEL_PARAMS, **(params or {})}
    num_classes = 1 if binary else int(max(y_train.max(), y_val.max()) + 1)
    model = build_hybrididsnet(
        input_shape=(X_train.shape[1], 1), num_classes=num_classes, binary=binary,
        cnn_filters=params["cnn_filters"], kernel_size=params["kernel_size"],
        lstm_units=params["lstm_units"], dropout=params["dropout"],
        dense_units=params["dense_units"], learning_rate=params["learning_rate"], l2_reg=params["l2_reg"]
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODEL_DIR / f"{run_name}.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=params["patience"], restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(str(METRICS_DIR / f"{run_name}_history.csv")),
    ]
    history = model.fit(
        to_sequence(X_train), y_train,
        validation_data=(to_sequence(X_val), y_val),
        epochs=params["epochs"], batch_size=params["batch_size"], callbacks=callbacks, verbose=2
    )
    pd.DataFrame(history.history).to_csv(METRICS_DIR / f"{run_name}_history_final.csv", index=False)
    return model, history
