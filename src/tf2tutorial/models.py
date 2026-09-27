"""Reusable model builders for the teaching chapters.

The builders accept only NumPy-friendly arguments and use ``tf.keras`` under
the hood. If TF is missing the import will fail loudly — these builders are
*not* used by chapter 01, which is NumPy-only.
"""

from __future__ import annotations

from typing import Sequence

# These helpers are imported lazily inside each function so that the package
# is importable even when TF is not installed.


def build_mlp(input_shape, num_classes: int, hidden: Sequence[int] = (128,)):
    """Simple fully-connected classifier."""
    from tensorflow.keras import layers, models  # type: ignore

    model = models.Sequential(name="mlp")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())
    for h in hidden:
        model.add(layers.Dense(h, activation="relu"))
    activation = "softmax" if num_classes > 2 else "sigmoid"
    units = num_classes if num_classes > 2 else 1
    model.add(layers.Dense(units, activation=activation))
    return model


def build_cnn(input_shape=(32, 32, 3), num_classes: int = 10):
    """A small VGG-style CNN suitable for CIFAR-10 on CPU."""
    from tensorflow.keras import layers, models  # type: ignore

    model = models.Sequential(name="cnn_small")
    model.add(layers.Input(shape=input_shape))
    for filters in (32, 64, 128):
        model.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))
        model.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model


def build_text_classifier(vocab_size: int, embed_dim: int = 64, num_classes: int = 2):
    """Embedding + GlobalAveragePooling1D text classifier (IMDB-style)."""
    from tensorflow.keras import layers, models  # type: ignore

    model = models.Sequential(name="text_classifier")
    model.add(layers.Input(shape=(None,)))
    model.add(layers.Embedding(vocab_size, embed_dim))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))
    activation = "sigmoid" if num_classes == 2 else "softmax"
    units = 1 if num_classes == 2 else num_classes
    model.add(layers.Dense(units, activation=activation))
    return model


def build_transfer_model(
    base_trainable: bool = False,
    image_size: int = 160,
    num_classes: int = 5,
    dropout: float = 0.2,
):
    """MobileNetV2 base + classification head (transfer learning)."""
    from tensorflow.keras import layers, models  # type: ignore
    from tensorflow.keras.applications import MobileNetV2  # type: ignore

    base = MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = base_trainable

    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="transfer_mobilenetv2")
