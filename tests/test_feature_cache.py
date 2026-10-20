from argparse import Namespace
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.integration
def test_frozen_head_cache_exports_real_network(tmp_path, monkeypatch):
    tf = pytest.importorskip("tensorflow")
    from PIL import Image
    from src.model import build_model
    import src.feature_cache as cache
    tf.keras.utils.set_random_seed(17)
    monkeypatch.setattr(cache, "ROOT", tmp_path)
    model, _ = build_model("efficientnet", 2, 32, weights=None)
    rows = []
    for index in range(8):
        path = tmp_path / f"image{index}.png"
        Image.fromarray(np.random.default_rng(index).integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(path)
        rows.append({"path": str(path), "class_index": index % 2, "split": "validation" if index >= 6 else "train"})
    augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal", seed=17)])
    output = tmp_path / "model"
    output.mkdir()
    args = Namespace(image_size=32, batch_size=4, feature_views=2, cache_features="tensorflow", seed=17, threads=None, learning_rate=.001, epochs=1)
    metadata = {"classes": ["a", "b"], "manifest_sha256": "fixture"}
    history = cache.train_cached_head(model, rows, metadata, args, augmentation, output)
    assert len(history.history["val_loss"]) == 1
    restored = tf.keras.models.load_model(output / "best.keras", compile=False)
    assert np.allclose(model(np.zeros((1, 32, 32, 3)), training=False), restored(np.zeros((1, 32, 32, 3)), training=False))
    assert (output / "feature_cache.json").exists()
