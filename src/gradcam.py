"""Grad-CAM using the actual flattened CNN/EfficientNet computation graph."""
import numpy as np
from PIL import Image
from src.preprocessing import preprocess_image, load_image


def gradcam(predictor, image, class_index=None, alpha=.4):
    import tensorflow as tf
    import matplotlib
    settings = predictor.settings
    pixels = preprocess_image(image, settings["image_size"])
    feature_model = tf.keras.Model(predictor.model.input, [predictor.model.get_layer(settings["cam_layer"]).output, predictor.model.get_layer("embedding").output, predictor.model.output])
    classifier = predictor.model.get_layer("classifier")
    with tf.GradientTape() as tape:
        features, embedding, probabilities = feature_model(pixels[None], training=False)
        index = int(tf.argmax(probabilities[0])) if class_index is None else int(class_index)
        if not 0 <= index < len(predictor.classes):
            raise ValueError("Invalid Grad-CAM class index.")
        # Pre-softmax score avoids saturation; dropout is disabled at inference.
        score = tf.reduce_sum(embedding * classifier.kernel[:, index], axis=-1) + classifier.bias[index]
    gradients = tape.gradient(score, features)
    if gradients is None:
        raise RuntimeError("The selected feature layer is disconnected from the classifier.")
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heat = tf.nn.relu(tf.reduce_sum(features * weights[:, None, None, :], axis=-1))[0].numpy()
    maximum = float(heat.max())
    if maximum <= 1e-12:
        raise ValueError("No positive class influence was found; no informative Grad-CAM can be displayed.")
    heat /= maximum
    original = load_image(image)
    resized = np.asarray(Image.fromarray(heat.astype(np.float32), mode="F").resize(original.size, Image.Resampling.BILINEAR))
    colors = matplotlib.colormaps["inferno"](resized)[..., :3] * 255
    overlay = np.clip((1 - alpha) * np.asarray(original) + alpha * colors, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay), heat
