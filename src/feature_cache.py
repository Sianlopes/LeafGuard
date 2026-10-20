"""Reproducible frozen EfficientNet embeddings and optional CPU acceleration.

Only backbone forward passes may use OpenVINO. The classification head is
trained with TensorFlow/Keras and exported in the original full Keras network.
No test label is used to fit a parameter. Cached augmentation is training-only.
"""
import hashlib
import os
from pathlib import Path
import time
import numpy as np
from src.common import ROOT, read_json, write_json, sha256
from src.dataset import tf_dataset


def embedding_fingerprint(model):
    digest = hashlib.sha256()
    for layer in model.layers:
        if layer.name in {"classifier", "head_dropout"}:
            continue
        digest.update(layer.name.encode())
        for weight in layer.get_weights():
            digest.update(str(weight.shape).encode())
            digest.update(weight.tobytes())
    return digest.hexdigest()


class FeatureExtractor:
    def __init__(self, model, directory, backend="tensorflow", threads=6):
        import tensorflow as tf
        self.model = tf.keras.Model(model.input, model.get_layer("embedding").output)
        self.backend = backend
        self.agreement = None
        if backend == "openvino":
            os.environ.setdefault("OV_TELEMETRY_DISABLED", "1")
            try:
                import openvino as ov
            except ImportError as exc:
                raise ValueError("Install requirements-training.txt for --cache-features openvino, or use tensorflow.") from exc
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            xml = directory / "embedding.xml"
            if not xml.exists():
                exported = directory / "saved_model"
                self.model.export(str(exported), verbose=False)
                converted = ov.convert_model(str(exported))
                ov.save_model(converted, xml, compress_to_fp16=False)
            self.compiled = ov.Core().compile_model(str(xml), "CPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": threads, "INFERENCE_PRECISION_HINT": "f32"})
            self.output = self.compiled.output(0)
            # Exact same pixels through both runtimes; fail instead of silently
            # training on a changed preprocessing or conversion contract.
            size = model.input_shape[1]
            probe = np.random.default_rng(42).uniform(0, 255, (2, size, size, 3)).astype(np.float32)
            reference = np.asarray(self.model(probe, training=False))
            accelerated = self(probe)
            max_error = float(np.abs(reference - accelerated).max())
            self.agreement = {"maximum_absolute_error": max_error, "mean_absolute_error": float(np.abs(reference - accelerated).mean()), "rtol": .002, "atol": .002}
            if not np.allclose(reference, accelerated, rtol=.002, atol=.002):
                raise ValueError(f"OpenVINO conversion failed the TensorFlow agreement check: {self.agreement}")
            print(f"Accelerator agreement: {self.agreement}", flush=True)

    def __call__(self, pixels):
        pixels = np.asarray(pixels, dtype=np.float32)
        if self.backend == "openvino":
            return np.asarray(self.compiled([pixels])[self.output]).copy()
        return np.asarray(self.model(pixels, training=False))


def cached_features(model, rows, metadata, image_size, batch_size, augmentation, views=2, backend="tensorflow", seed=42, threads=6):
    fingerprint = embedding_fingerprint(model)
    directory = ROOT / "data/cache/features" / f"{metadata['manifest_sha256'][:12]}-{fingerprint[:12]}-{image_size}-v{views}-{seed}-{backend}"
    directory.mkdir(parents=True, exist_ok=True)
    contract = {"manifest_sha256": metadata["manifest_sha256"], "embedding_fingerprint": fingerprint, "image_size": image_size, "views": views, "seed": seed, "backend": backend, "images": len(rows), "dimension": model.get_layer("embedding").output.shape[-1], "augmentation": "flip-horizontal-vertical,rotation-.08,zoom-.1,contrast-.1,clip-0-255"}
    state_path = directory / "state.json"
    train_rows = [row for row in rows if row["split"] == "train"]
    phase_rows = {"original": rows, **{f"augmented_{v}": train_rows for v in range(1, views)}}
    state = read_json(state_path) if state_path.exists() else {"contract": contract, "completed": {phase: 0 for phase in phase_rows}, "complete": False}
    if state["contract"] != contract:
        raise ValueError("Feature-cache configuration mismatch. Use a new cache directory.")
    arrays = {}
    for phase, selected in phase_rows.items():
        path = directory / f"{phase}.npy"
        if path.exists():
            array = np.load(path, mmap_mode="r+")
            if array.shape != (len(selected), contract["dimension"]):
                raise ValueError("Feature cache has an invalid shape.")
        else:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(selected), contract["dimension"]))
        arrays[phase] = array
    if not state["complete"]:
        extractor = FeatureExtractor(model, directory / "accelerator", backend, threads)
        state["accelerator_agreement"] = extractor.agreement
        for layer in augmentation.layers:
            saved_seed = state.get("augmentation_seed_states", {}).get(layer.name)
            if saved_seed is not None:
                layer.seed_generator.state.assign(saved_seed)
        for phase, selected in phase_rows.items():
            start = state["completed"][phase]
            if start == len(selected):
                continue
            dataset = tf_dataset(selected[start:], image_size, batch_size)
            completed = start
            began = time.perf_counter()
            for batch_index, (images, _) in enumerate(dataset):
                if phase != "original":
                    import tensorflow as tf
                    images = tf.clip_by_value(augmentation(images, training=True), 0., 255.)
                values = extractor(images)
                if not np.isfinite(values).all():
                    raise ValueError("Feature extractor returned non-finite values.")
                arrays[phase][completed:completed + len(values)] = values
                completed += len(values)
                if batch_index % 25 == 0 or completed == len(selected):
                    arrays[phase].flush()
                    state["completed"][phase] = completed
                    state["augmentation_seed_states"] = {layer.name: np.asarray(layer.seed_generator.state).tolist() for layer in augmentation.layers if hasattr(layer, "seed_generator")}
                    write_json(state_path, state)
                    rate = (completed - start) / max(time.perf_counter() - began, .001)
                    print(f"Features {phase}: {completed:,}/{len(selected):,}; {rate:.1f} images/s", flush=True)
        state["complete"] = True
        state["files"] = {phase: {"name": f"{phase}.npy", "sha256": sha256(directory / f"{phase}.npy")} for phase in arrays}
        write_json(state_path, state)
    return arrays, directory, state


def train_cached_head(model, rows, metadata, args, augmentation, output):
    import tensorflow as tf
    from src.model import compile_model
    from src.training_utils import callbacks
    arrays, directory, state = cached_features(model, rows, metadata, args.image_size, args.batch_size, augmentation, args.feature_views, args.cache_features, args.seed, args.threads or 6)
    training = np.array([i for i, row in enumerate(rows) if row["split"] == "train"])
    validation = np.array([i for i, row in enumerate(rows) if row["split"] == "validation"])
    labels = np.array([int(row["class_index"]) for row in rows], dtype=np.int32)
    x_train = np.concatenate([np.asarray(arrays["original"][training])] + [np.asarray(arrays[f"augmented_{v}"]) for v in range(1, args.feature_views)])
    y_train = np.tile(labels[training], args.feature_views)
    inputs = tf.keras.Input((x_train.shape[1],))
    features = tf.keras.layers.Dropout(.4)(inputs)
    outputs = tf.keras.layers.Dense(len(metadata["classes"]), activation="softmax")(features)
    head = tf.keras.Model(inputs, outputs)
    head.layers[-1].set_weights(model.get_layer("classifier").get_weights())
    compile_model(head, args.learning_rate)
    class_weights = None
    if getattr(args, "class_weight", "none") == "balanced":
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = dict(enumerate(compute_class_weight("balanced", classes=np.arange(len(metadata["classes"])), y=y_train)))
    checkpoint_dir = Path(output) / "head"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = head.fit(x_train, y_train, validation_data=(np.asarray(arrays["original"][validation]), labels[validation]), epochs=args.epochs, batch_size=args.batch_size, shuffle=True, class_weight=class_weights, callbacks=callbacks(checkpoint_dir), verbose=2)
    best = tf.keras.models.load_model(checkpoint_dir / "best.keras", compile=False)
    model.get_layer("classifier").set_weights(best.layers[-1].get_weights())
    compile_model(model, args.learning_rate)
    model.save(Path(output) / "best.keras")
    write_json(Path(output) / "feature_cache.json", {"directory": str(directory), "contract": state["contract"], "files": state["files"], "accelerator_agreement": state.get("accelerator_agreement"), "classifier_training": "TensorFlow/Keras on fixed pretrained embeddings", "fine_tuning": False})
    return history


def features_for_rows(predictor, selected_rows):
    """Reuse only an exact checkpoint-backbone + manifest-matching cache."""
    pointer = predictor.directory / "feature_cache.json"
    if not pointer.exists():
        return None
    details = read_json(pointer)
    directory = ROOT / Path(details["directory"])
    path = directory / "original.npy"
    if not path.exists():
        return None  # Colab packages intentionally omit this optional cache.
    if details["contract"]["manifest_sha256"] != predictor.settings["manifest_sha256"] or details["contract"]["embedding_fingerprint"] != embedding_fingerprint(predictor.model):
        raise ValueError("Feature cache does not match this checkpoint's backbone.")
    from src.dataset import load_manifest
    all_rows, _, _ = load_manifest(ROOT / Path(predictor.settings.get("processed_directory", "data/processed")))
    indices = {row["path"]: i for i, row in enumerate(all_rows)}
    if sha256(path) != details["files"]["original"]["sha256"]:
        raise ValueError("Feature-cache checksum mismatch.")
    array = np.load(path, mmap_mode="r")
    return np.asarray(array[[indices[row["path"]] for row in selected_rows]])


def predict_from_features(predictor, features):
    import tensorflow as tf
    classifier = predictor.model.get_layer("classifier")
    return np.asarray(tf.nn.softmax(tf.linalg.matmul(features, classifier.kernel) + classifier.bias))
