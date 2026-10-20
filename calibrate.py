"""Calibrate on reserved leaf groups and validate rejection on independent OOD images.

OOD tune/audit directories must contain unrelated objects AND unsupported plants.
Use separate top-level folders for these categories. Document capture-session
independence when available, or its absence. Every OOD image is a negative
example. Do not put PlantVillage test images in these directories.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from src.common import write_json, sha256
from src.dataset import load_manifest, tf_dataset
from src.preprocessing import load_image, preprocess_image
from src.calibration import fit_temperature, temperature_scale, normalize_embeddings, embedding_distances, ece, choose_threshold, rejection_statistics


def partition_groups(rows, seed):
    rng = np.random.default_rng(seed)
    groups = sorted({row["group_id"] for row in rows})
    rng.shuffle(groups)
    if len(groups) < 9:
        raise ValueError("Calibration needs at least nine independent leaf groups; use the full dataset.")
    partitions = np.array_split(groups, 3)
    return [[row for row in rows if row["group_id"] in set(part)] for part in partitions]


def collect(predictor, rows, batch_size):
    from src.feature_cache import features_for_rows, predict_from_features
    cached = features_for_rows(predictor, rows)
    if cached is not None:
        return predict_from_features(predictor, cached), cached, np.array([int(row["class_index"]) for row in rows])
    dataset = tf_dataset(rows, predictor.settings["image_size"], batch_size)
    probs, embeddings = predictor.joint_model.predict(dataset, verbose=1)
    return np.asarray(probs), np.asarray(embeddings), np.array([int(row["class_index"]) for row in rows])


def load_ood(directory, predictor, batch_size, forbidden):
    paths = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise ValueError(f"No OOD images in {directory}")
    probabilities, features, hashes, groups = [], [], [], []
    batch = []
    for path in paths:
        image = load_image(path)
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest in forbidden:
            raise ValueError(f"Duplicate or in-distribution image in OOD data: {path}")
        forbidden.add(digest)
        hashes.append(digest)
        relative = path.relative_to(Path(directory))
        groups.append(relative.parts[0] if len(relative.parts) > 1 else "unspecified")
        batch.append(preprocess_image(image, predictor.settings["image_size"]))
        if len(batch) == batch_size or path == paths[-1]:
            p, f = predictor.joint_model(np.stack(batch), training=False)
            probabilities.extend(np.asarray(p))
            features.extend(np.asarray(f))
            batch = []
    return np.asarray(probabilities), np.asarray(features), hashes, np.array(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/efficientnet")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--ood-dir", required=True, help="OOD tuning images, independently collected")
    parser.add_argument("--ood-audit-dir", required=True, help="Independent OOD audit images; never used to choose thresholds")
    parser.add_argument("--ood-description", required=True, help="Document sources, unsupported species, unrelated objects and capture-session separation")
    parser.add_argument("--target-accuracy", type=float, default=.9)
    parser.add_argument("--max-ood-acceptance", type=float, default=.05)
    parser.add_argument("--min-audit-images", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if not .5 <= args.target_accuracy <= 1 or not 0 <= args.max_ood_acceptance < 1 or args.min_audit_images < 30 or args.batch_size < 1:
        parser.error("Invalid calibration parameters (audit minimum must be at least 30).")
    try:
        from src.training_utils import set_seed
        set_seed(args.seed, args.cpu, args.threads)
        from src.inference import Predictor
        predictor = Predictor(args.model_dir)
        rows, classes, metadata = load_manifest(args.data_dir)
        if classes != predictor.classes or metadata["manifest_sha256"] != predictor.settings["manifest_sha256"]:
            raise ValueError("Calibration dataset does not match the trained experiment.")
        fit_rows, tune_rows, audit_rows = partition_groups([r for r in rows if r["split"] == "calibration"], args.seed)
        # Prototypes use training images only; aggregate batches to avoid retaining all embeddings.
        train_rows = [r for r in rows if r["split"] == "train"]
        dimension = predictor.joint_model.output_shape[1][-1]
        sums, counts = np.zeros((len(classes), dimension)), np.zeros(len(classes))
        from src.feature_cache import features_for_rows
        cached = features_for_rows(predictor, train_rows)
        if cached is None:
            batches = ((predictor.joint_model(images, training=False)[1], np.asarray(labels)) for images, labels in tf_dataset(train_rows, predictor.settings["image_size"], args.batch_size))
        else:
            training_labels = np.array([int(row["class_index"]) for row in train_rows])
            batches = ((cached[start:start + args.batch_size], training_labels[start:start + args.batch_size]) for start in range(0, len(train_rows), args.batch_size))
        for features, labels in batches:
            for feature, label in zip(normalize_embeddings(features), np.asarray(labels)):
                sums[label] += feature
                counts[label] += 1
        if (counts == 0).any():
            raise ValueError("Missing training class for embedding prototypes.")
        prototypes = normalize_embeddings(sums / counts[:, None])
        fit_probs, _, fit_labels = collect(predictor, fit_rows, args.batch_size)
        temperature = fit_temperature(fit_probs, fit_labels)
        tune_raw, tune_features, tune_labels = collect(predictor, tune_rows, args.batch_size)
        audit_raw, audit_features, audit_labels = collect(predictor, audit_rows, args.batch_size)
        forbidden = {r["pixel_sha256"] for r in rows}
        ood_raw, ood_features, tune_hashes, tune_groups = load_ood(args.ood_dir, predictor, args.batch_size, forbidden)
        audit_ood_raw, audit_ood_features, audit_hashes, audit_groups = load_ood(args.ood_audit_dir, predictor, args.batch_size, forbidden)
        required_categories = {"unsupported_leaves", "unrelated_objects"}
        if not required_categories.issubset(set(tune_groups)) or not required_categories.issubset(set(audit_groups)):
            raise ValueError("Both OOD collections require unsupported_leaves/ and unrelated_objects/ subdirectories so each negative category is audited separately.")
        tune_probs, audit_probs, ood_probs, audit_ood_probs = [temperature_scale(p, temperature) for p in (tune_raw, audit_raw, ood_raw, audit_ood_raw)]
        tune_dist, audit_dist, ood_dist, audit_ood_dist = [embedding_distances(f, p.argmax(axis=1), prototypes) for f, p in ((tune_features, tune_probs), (audit_features, audit_probs), (ood_features, ood_probs), (audit_ood_features, audit_ood_probs))]
        chosen = choose_threshold(tune_probs, tune_dist, tune_labels, ood_probs, ood_dist, args.target_accuracy, args.max_ood_acceptance, tune_groups)
        feasible = chosen is not None
        chosen = chosen or {"threshold": 1., "distance_threshold": -1., "coverage": 0.}
        known = rejection_statistics(audit_probs, audit_dist, audit_labels, chosen["threshold"], chosen["distance_threshold"])
        unknown = rejection_statistics(audit_ood_probs, audit_ood_dist, None, chosen["threshold"], chosen["distance_threshold"])
        unknown_groups = {str(group): rejection_statistics(audit_ood_probs[audit_groups == group], audit_ood_dist[audit_groups == group], None, chosen["threshold"], chosen["distance_threshold"]) for group in np.unique(audit_groups)}
        group_audit_passed = all(stats["images"] >= 30 and stats["coverage"] <= args.max_ood_acceptance for stats in unknown_groups.values())
        validated = bool(feasible and known["images"] >= args.min_audit_images and unknown["images"] >= args.min_audit_images and known["accepted"] >= 30 and (known["accepted_accuracy"] or 0) >= args.target_accuracy and unknown["coverage"] <= args.max_ood_acceptance and group_audit_passed)
        out = Path(args.model_dir)
        np.savez_compressed(out / "prototypes.npz", prototypes=prototypes)
        result = {"temperature": temperature, **chosen, "validated": validated, "model_sha256": predictor.settings["model_sha256"], "manifest_sha256": metadata["manifest_sha256"], "prototypes_sha256": sha256(out / "prototypes.npz"), "target_accuracy": args.target_accuracy, "max_ood_acceptance": args.max_ood_acceptance, "audit_id": known, "audit_ood": unknown, "audit_ece_before": ece(audit_raw, audit_labels), "audit_ece_after": ece(audit_probs, audit_labels), "fit_images": len(fit_rows), "tune_images": len(tune_rows), "ood_description": args.ood_description, "ood_tune_hashes": tune_hashes, "ood_audit_hashes": audit_hashes, "seed": args.seed, "limitations": "Empirical audit on supplied images, not proof of universal OOD rejection. Audit cannot be reused to tune thresholds; collect a fresh audit set after changes."}
        result["audit_ood_groups"] = unknown_groups
        result["complete_physical_leaf_verification"] = metadata.get("complete_physical_leaf_verification", True)
        result["split_limitation"] = metadata.get("limitations")
        write_json(out / "calibration.json", result)
        print(f"Rejection audit passed: {validated}; ID audit {known}; OOD audit {unknown}. Calibration saved to {out}.")
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(1, f"Calibration error: {exc}\n")


if __name__ == "__main__":
    main()
