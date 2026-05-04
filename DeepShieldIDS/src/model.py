
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model


class AttentionPooling(layers.Layer):
    """Trainable attention pooling over time steps."""
    def __init__(self, attention_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.score_dense = layers.Dense(attention_dim, activation="tanh")
        self.weight_dense = layers.Dense(1)

    def call(self, inputs, return_attention=False):
        scores = self.weight_dense(self.score_dense(inputs))
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(inputs * weights, axis=1)
        if return_attention:
            return context, tf.squeeze(weights, axis=-1)
        return context

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"attention_dim": self.attention_dim})
        return cfg


def build_hybrididsnet(
    input_shape: tuple[int, int],
    num_classes: int = 1,
    binary: bool = True,
    cnn_filters: int = 64,
    kernel_size: int = 3,
    lstm_units: int = 128,
    dropout: float = 0.3,
    dense_units: int = 128,
    learning_rate: float = 1e-3,
    l2_reg: float = 1e-3,
):
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=input_shape, name="traffic_features")
    x = layers.Conv1D(cnn_filters, kernel_size, padding="same", activation="relu", kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=reg))(x)
    x = layers.Dropout(dropout)(x)
    x = AttentionPooling(attention_dim=64, name="attention_pooling")(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    if binary:
        outputs = layers.Dense(1, activation="sigmoid", name="binary_output")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="multiclass_output")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    model = Model(inputs, outputs, name="HybridIDSNet")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss, metrics=metrics)
    return model


def to_sequence(X):
    import numpy as np
    arr = X.values if hasattr(X, "values") else X
    return np.asarray(arr, dtype="float32")[..., None]
