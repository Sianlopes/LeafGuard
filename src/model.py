"""Serializable Keras networks with embedded architecture-specific scaling."""
import tensorflow as tf
from tensorflow import keras


def build_model(architecture, num_classes, image_size=224, weights="imagenet", cnn_variant="standard"):
    if num_classes < 2:
        raise ValueError("At least two actual dataset classes are required.")
    inputs = keras.Input((image_size, image_size, 3), name="rgb_pixels")
    if architecture == "efficientnet":
        # Keras EfficientNet includes Rescaling/Normalization: do NOT divide by 255 externally.
        try:
            backbone = keras.applications.EfficientNetB0(include_top=False, weights=weights, input_tensor=inputs)
        except Exception as exc:
            raise RuntimeError("Unable to load ImageNet weights. Restore network/cache or explicitly use --weights none for an unpretrained experiment.") from exc
        for layer in backbone.layers:
            layer.trainable = False
        x = backbone.output
        cam_layer = "top_activation"
    elif architecture == "cnn":
        x = keras.layers.Rescaling(1.0 / 255, name="cnn_rescaling")(inputs)
        if cnn_variant not in {"standard", "compact"}:
            raise ValueError("Unknown CNN variant.")
        channels = (16, 32, 64) if cnn_variant == "compact" else (32, 64, 128)
        for index, filters in enumerate(channels):
            if cnn_variant == "compact" and index > 0:
                x = keras.layers.SeparableConv2D(filters, 3, padding="same", use_bias=False, name=f"conv_{index}")(x)
            else:
                x = keras.layers.Conv2D(filters, 3, strides=2 if cnn_variant == "compact" and index == 0 else 1, padding="same", use_bias=False, name=f"conv_{index}")(x)
            x = keras.layers.BatchNormalization(name=f"bn_{index}")(x)
            x = keras.layers.ReLU(name=f"relu_{index}")(x)
            x = keras.layers.MaxPooling2D(name=f"pool_{index}")(x)
            x = keras.layers.Dropout(.15, name=f"dropout_{index}")(x)
        x = keras.layers.Conv2D(96 if cnn_variant == "compact" else 192, 1 if cnn_variant == "compact" else 3, padding="same", activation="relu", name="cam_features")(x)
        cam_layer = "cam_features"
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    x = keras.layers.GlobalAveragePooling2D(name="embedding")(x)
    x = keras.layers.Dropout(.4, name="head_dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = keras.Model(inputs, outputs, name=f"leafguard_{architecture}")
    return model, cam_layer


def compile_model(model, learning_rate):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def enable_fine_tuning(model, count=30):
    head_names = {"embedding", "head_dropout", "classifier"}
    candidates = [layer for layer in model.layers if layer.name not in head_names and not isinstance(layer, keras.layers.InputLayer)]
    for layer in candidates:
        layer.trainable = False
    for layer in candidates[-count:] if count > 0 else []:
        # Preserve pretrained moving statistics, particularly for small batches.
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True
