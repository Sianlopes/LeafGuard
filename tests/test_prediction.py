import numpy as np
import pytest
from PIL import Image
from src.inference import Predictor, ModelUnavailableError, disease_information
from src.calibration import temperature_scale, fit_temperature, embedding_distances, choose_threshold
from src.common import write_json, sha256


def test_missing_model_has_actionable_error(tmp_path):
    with pytest.raises(ModelUnavailableError, match="train.py"):
        Predictor(tmp_path)


def test_unknown_advice_is_neutral():
    info = disease_information("Unknown___Unverified")
    assert not info["prevention"]
    assert "not available" in info["description"]


def test_temperature_and_distance():
    original = np.array([[.99, .01], [.8, .2], [.6, .4], [.05, .95]])
    labels = np.array([0, 1, 0, 1])
    temperature = fit_temperature(original, labels)
    assert temperature > 0
    probabilities = temperature_scale(original, temperature)
    assert np.allclose(probabilities.sum(axis=1), 1.)
    assert np.array_equal(original.argmax(axis=1), probabilities.argmax(axis=1))
    assert np.allclose(embedding_distances([[1., 0.], [-1., 0.]], [0, 0], np.array([[1., 0.]])), [0, 2])


def test_no_feasible_rejection_rule():
    probabilities = np.array([[.99, .01]] * 10)
    assert choose_threshold(probabilities, np.zeros(10), np.zeros(10), probabilities, np.zeros(10)) is None


@pytest.mark.integration
def test_real_cnn_training_serialization_inference_and_gradcam(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from src.model import build_model, compile_model
    from src.gradcam import gradcam
    tf.keras.utils.set_random_seed(42)
    model, cam = build_model("cnn", 2, 32)
    compile_model(model, .001)
    # Synthetic FIXTURES verify execution, not plant disease performance.
    pixels = np.random.default_rng(42).uniform(0, 255, (4, 32, 32, 3)).astype(np.float32)
    model.train_on_batch(pixels, np.array([0, 1, 0, 1]))
    model.save(tmp_path / "best.keras")
    write_json(tmp_path / "class_names.json", ["Apple___healthy", "Potato___Early_blight"])
    write_json(tmp_path / "inference_config.json", {"schema_version": 1, "architecture": "cnn", "image_size": 32, "color_mode": "RGB", "pixel_range": [0, 255], "resize": "PIL_BILINEAR", "scaling": "embedded_in_model", "embedding_layer": "embedding", "cam_layer": cam, "uncertainty_threshold": .8, "manifest_sha256": "test-fixture", "classes_sha256": sha256(tmp_path / "class_names.json"), "model_sha256": sha256(tmp_path / "best.keras")})
    predictor = Predictor(tmp_path)
    image = Image.fromarray(pixels[0].astype(np.uint8))
    result = predictor.predict(image)
    assert result["uncertain"] and result["plant"] is None
    assert len(result["top_three"]) == 2
    assert np.isclose(sum(c["confidence"] for c in result["top_three"]), 1.)
    # Real-network features exercise calibrated acceptance and distance rejection.
    probabilities, features = predictor.joint_model(np.asarray(image, dtype=np.float32)[None], training=False)
    from src.calibration import normalize_embeddings
    prototype = normalize_embeddings(np.asarray(features))[0]
    np.savez_compressed(tmp_path / "prototypes.npz", prototypes=np.stack([prototype, prototype]))
    calibration = {"temperature": 1., "threshold": 0., "distance_threshold": 2., "validated": True, "model_sha256": predictor.settings["model_sha256"], "manifest_sha256": "test-fixture", "prototypes_sha256": sha256(tmp_path / "prototypes.npz")}
    write_json(tmp_path / "calibration.json", calibration)
    accepted = Predictor(tmp_path).predict(image)
    assert not accepted["uncertain"] and accepted["plant"] is not None
    calibration["distance_threshold"] = -1.
    write_json(tmp_path / "calibration.json", calibration)
    rejected = Predictor(tmp_path).predict(image)
    assert rejected["uncertain"] and rejected["disease_info"] is None
    try:
        overlay, heat = gradcam(predictor, image)
        assert overlay.size == image.size and np.isfinite(heat).all()
    except ValueError as exc:
        assert "No positive class influence" in str(exc)
    # A tampered checkpoint cannot silently reuse its metadata.
    with (tmp_path / "best.keras").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ModelUnavailableError, match="checksum"):
        Predictor(tmp_path)


@pytest.mark.integration
def test_efficientnet_graph_and_gradcam(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from src.model import build_model, enable_fine_tuning, compile_model
    model, cam = build_model("efficientnet", 3, 32, weights=None)
    assert cam == "top_activation"
    assert not model.get_layer("top_conv").trainable
    enable_fine_tuning(model, 30)
    assert model.get_layer("top_conv").trainable
    assert not model.get_layer("top_bn").trainable
    compile_model(model, 1e-5)
    pixels = np.random.default_rng(3).uniform(0, 255, (2, 32, 32, 3)).astype(np.float32)
    model.train_on_batch(pixels, np.array([0, 1]))
    probe = tf.keras.Model(model.inputs, [model.get_layer(cam).output, model.output])
    with tf.GradientTape() as tape:
        features, probabilities = probe(pixels[:1], training=False)
        score = probabilities[:, 0]
    assert tape.gradient(score, features) is not None
