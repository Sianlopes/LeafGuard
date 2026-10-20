"""Train one architecture; checkpoint selection uses validation loss only."""
import argparse
import os
from pathlib import Path
from src.common import ROOT, config, write_json, sha256
from src.dataset import load_manifest, tf_dataset
from src.training_utils import set_seed, callbacks, save_history


def main():
    cfg = config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cnn", "efficientnet"], default="efficientnet")
    parser.add_argument("--data-dir", default=cfg["processed_path"])
    parser.add_argument("--output")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--cnn-variant", choices=["standard", "compact"], default=cfg.get("cnn_variant", "standard"))
    parser.add_argument("--cache-features", choices=["none", "tensorflow", "openvino"], default="none")
    parser.add_argument("--feature-views", type=int, default=2, help="Original plus training-only augmented views for a frozen feature cache")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    for name in ("epochs", "fine_tune_epochs", "batch_size", "image_size", "seed", "unfreeze_layers"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=cfg[name])
    for name in ("learning_rate", "fine_tune_learning_rate"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=cfg[name])
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32 or args.fine_tune_epochs < 0 or args.unfreeze_layers < 0 or min(args.learning_rate, args.fine_tune_learning_rate) <= 0:
        parser.error("Invalid training hyperparameters.")
    if args.feature_views < 1 or (args.threads is not None and args.threads < 1):
        parser.error("Feature views and thread count must be positive.")
    if args.cache_features != "none" and (args.model != "efficientnet" or args.fine_tune_epochs):
        parser.error("Frozen feature caching requires --model efficientnet --fine-tune-epochs 0.")
    out = Path(args.output or f"models/{args.model}")
    if (out / "best.keras").exists():
        parser.error(f"{out}/best.keras already exists. Use a new --output directory to preserve the experiment.")
    try:
        rows, classes, metadata = load_manifest(args.data_dir)
        os.environ.setdefault("KERAS_HOME", str(ROOT / "data/cache/keras"))
        set_seed(args.seed, args.cpu, args.threads)
        import tensorflow as tf
        from src.model import build_model, compile_model, enable_fine_tuning
        out.mkdir(parents=True, exist_ok=True)
        model, cam_layer = build_model(args.model, len(classes), args.image_size, None if args.weights == "none" else args.weights, args.cnn_variant)
        train = tf_dataset([r for r in rows if r["split"] == "train"], args.image_size, args.batch_size, True, args.seed)
        validation = tf_dataset([r for r in rows if r["split"] == "validation"], args.image_size, args.batch_size)
        # Augmentation only occurs in the training dataset, never validation/test/inference.
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=args.seed),
            tf.keras.layers.RandomRotation(.08, seed=args.seed + 1),
            tf.keras.layers.RandomZoom(.1, seed=args.seed + 2),
            tf.keras.layers.RandomContrast(.1, seed=args.seed + 3),
        ])
        # Overlap one augmented batch with optimization without unbounded buffers.
        train = train.map(lambda x, y: (tf.clip_by_value(augmentation(x, training=True), 0., 255.), y)).prefetch(1)
        compile_model(model, args.learning_rate)
        class_weights = None
        if args.class_weight == "balanced":
            import numpy as np
            from sklearn.utils.class_weight import compute_class_weight
            y = np.array([int(row["class_index"]) for row in rows if row["split"] == "train"])
            class_weights = dict(enumerate(compute_class_weight("balanced", classes=np.arange(len(classes)), y=y)))
        print(f"Devices: {tf.config.list_physical_devices()}\nClasses: {len(classes)}")
        if args.cache_features == "none":
            histories = [model.fit(train, validation_data=validation, epochs=args.epochs, callbacks=callbacks(out), class_weight=class_weights, shuffle=False, verbose=2)]
        else:
            from src.feature_cache import train_cached_head
            histories = [train_cached_head(model, rows, metadata, args, augmentation, out)]
        best = min(histories[0].history["val_loss"])
        if args.model == "efficientnet" and args.fine_tune_epochs and args.unfreeze_layers:
            model = tf.keras.models.load_model(out / "best.keras")
            enable_fine_tuning(model, args.unfreeze_layers)
            compile_model(model, args.fine_tune_learning_rate)
            histories.append(model.fit(train, validation_data=validation, epochs=args.fine_tune_epochs, callbacks=callbacks(out, best, "fine_tune"), class_weight=class_weights, shuffle=False, verbose=2))
        history = save_history(out, histories)
        write_json(out / "class_names.json", classes)
        write_json(out / "inference_config.json", {"schema_version": 1, "architecture": args.model, "cnn_variant": args.cnn_variant if args.model == "cnn" else None, "image_size": args.image_size, "color_mode": "RGB", "pixel_range": [0, 255], "resize": "PIL_BILINEAR", "scaling": "embedded_in_model", "cam_layer": cam_layer, "embedding_layer": "embedding", "uncertainty_threshold": cfg["uncertainty_threshold"], "manifest_sha256": metadata["manifest_sha256"], "processed_directory": str(Path(args.data_dir).resolve()), "classes_sha256": sha256(out / "class_names.json"), "model_sha256": sha256(out / "best.keras")})
        write_json(out / "training_config.json", {**vars(args), "tensorflow_version": tf.__version__, "devices": [str(d) for d in tf.config.list_physical_devices()], "pretrained": args.model == "efficientnet" and args.weights == "imagenet", "dataset": metadata, "head_epochs_completed": len(histories[0].epoch), "fine_tune_epochs_completed": len(histories[1].epoch) if len(histories) > 1 else 0})
        from src.evaluation import plot_history
        plot_history(history, Path(args.reports_dir) / "figures" / args.model)
        print(f"Best checkpoint and inference metadata saved in {out}. Test set was not used.")
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(1, f"Training error: {exc}\n")


if __name__ == "__main__":
    main()
