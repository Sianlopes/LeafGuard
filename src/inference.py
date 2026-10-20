"""Actual model inference; missing, mismatched or uncalibrated artifacts fail safely."""
from pathlib import Path
from functools import lru_cache
import numpy as np
from src.common import ROOT, read_json, sha256, parse_label, model_directory
from src.preprocessing import preprocess_image, validate_preprocessing
from src.calibration import temperature_scale, embedding_distances

UNCERTAIN_MESSAGE = "Unable to identify this leaf reliably. Try uploading a clearer image or a leaf from a supported plant species."


class ModelUnavailableError(RuntimeError):
    pass


def disease_information(label):
    entries = read_json(ROOT / "data/disease_info.json")
    plant, disease = parse_label(label)
    if disease.lower() == "healthy":
        return entries["healthy"]
    return entries.get(label, {"disease_name": disease, "description": "Curated guidance is not available for this category. Confirm identification with a local agricultural extension specialist.", "symptoms": [], "prevention": [], "sources": []})


class Predictor:
    def __init__(self, directory=None):
        self.directory = Path(directory or model_directory()).resolve()
        required = ["best.keras", "class_names.json", "inference_config.json"]
        missing = [name for name in required if not (self.directory / name).is_file()]
        if missing:
            raise ModelUnavailableError(f"Trained model unavailable in {self.directory}. Missing: {', '.join(missing)}. Run train.py or copy the complete Colab model directory here.")
        try:
            self.classes = read_json(self.directory / "class_names.json")
            self.settings = read_json(self.directory / "inference_config.json")
            validate_preprocessing(self.settings)
            if self.settings.get("schema_version") != 1 or self.settings.get("scaling") != "embedded_in_model":
                raise ValueError("Unsupported artifact configuration version.")
            if not isinstance(self.classes, list) or len(self.classes) < 2 or len(set(self.classes)) != len(self.classes) or not all(isinstance(c, str) and "___" in c for c in self.classes):
                raise ValueError("Invalid saved class mapping.")
            if sha256(self.directory / "best.keras") != self.settings["model_sha256"]:
                raise ValueError("Saved model checksum does not match inference metadata.")
            if sha256(self.directory / "class_names.json") != self.settings["classes_sha256"]:
                raise ValueError("Class mapping checksum does not match inference metadata.")
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.directory / "best.keras", compile=False)
            if self.model.output_shape[-1] != len(self.classes) or tuple(self.model.input_shape[1:]) != (self.settings["image_size"], self.settings["image_size"], 3):
                raise ValueError("Model dimensions do not match saved metadata.")
            self.joint_model = tf.keras.Model(self.model.input, [self.model.output, self.model.get_layer(self.settings["embedding_layer"]).output])
            self.calibration = None
            self.prototypes = None
            calibration_path = self.directory / "calibration.json"
            if calibration_path.exists():
                calibration = read_json(calibration_path)
                if calibration.get("model_sha256") != self.settings["model_sha256"] or calibration.get("manifest_sha256") != self.settings["manifest_sha256"]:
                    raise ValueError("Stale calibration: recalibrate this exact trained checkpoint.")
                prototype_path = self.directory / "prototypes.npz"
                if sha256(prototype_path) != calibration["prototypes_sha256"]:
                    raise ValueError("Prototype checksum mismatch.")
                with np.load(prototype_path, allow_pickle=False) as archive:
                    self.prototypes = archive["prototypes"]
                if self.prototypes.shape != (len(self.classes), self.joint_model.output_shape[1][-1]) or not np.isfinite(self.prototypes).all():
                    raise ValueError("Invalid embedding prototypes.")
                self.calibration = calibration
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(f"Cannot load model artifacts: {exc}") from exc

    def predict(self, image, threshold=None):
        pixels = preprocess_image(image, self.settings["image_size"])
        raw, features = self.joint_model(pixels[None], training=False)
        raw, features = np.asarray(raw), np.asarray(features)
        if not np.isfinite(raw).all() or raw.shape != (1, len(self.classes)) or (raw < 0).any() or not np.allclose(raw.sum(axis=1), 1., atol=1e-4):
            raise RuntimeError("Model returned invalid probabilities.")
        calibrated = self.calibration is not None
        probabilities = temperature_scale(raw, self.calibration["temperature"] if calibrated else 1.)[0]
        indices = np.argsort(probabilities)[::-1][:3]
        best = int(indices[0])
        cutoff = self.calibration["threshold"] if calibrated else self.settings["uncertainty_threshold"]
        if threshold is not None:
            if not 0 <= threshold <= 1:
                raise ValueError("Threshold must be between zero and one.")
            cutoff = max(cutoff, threshold)
        distance = float(embedding_distances(features, [best], self.prototypes)[0]) if calibrated else None
        accepted = bool(calibrated and self.calibration.get("validated", False) and probabilities[best] >= cutoff and distance <= self.calibration["distance_threshold"])
        top = []
        for index in indices:
            plant, disease = parse_label(self.classes[index])
            top.append({"label": self.classes[index], "plant": plant, "disease": disease, "confidence": float(probabilities[index])})
        return {"plant": top[0]["plant"] if accepted else None, "disease": top[0]["disease"] if accepted else None, "label": top[0]["label"] if accepted else None, "confidence": float(probabilities[best]), "top_three": top, "uncertain": not accepted, "message": "AI-based suggestion; confirm with an agricultural specialist." if accepted else UNCERTAIN_MESSAGE, "calibrated": calibrated, "rejection_validated": bool(calibrated and self.calibration.get("validated")), "threshold": cutoff, "embedding_distance": distance, "disease_info": disease_information(self.classes[best]) if accepted else None, "predicted_index": best, "model": self.settings["architecture"]}


@lru_cache(maxsize=2)
def get_predictor(directory):
    return Predictor(directory)


def predict_leaf(image, model_dir=None, threshold=None):
    return get_predictor(str(Path(model_dir or model_directory()).resolve())).predict(image, threshold)
