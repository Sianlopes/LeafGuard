"""Held-out metrics and figures. No sample or placeholder scores."""
from pathlib import Path
import numpy as np
from src.common import write_json, read_json


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_history(history, output):
    plt = pyplot()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for key, title in (("accuracy", "Training accuracy"), ("val_accuracy", "Validation accuracy"), ("loss", "Training loss"), ("val_loss", "Validation loss")):
        if key not in history:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(history[key]) + 1), history[key], color="#26734d", linewidth=2)
        ax.set(xlabel="Epoch", ylabel=title, title=title)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output / f"{key}.png", dpi=160)
        plt.close(fig)


def compute_metrics(y_true, probabilities, classes):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    y_pred = np.asarray(probabilities).argmax(axis=1)
    report = classification_report(y_true, y_pred, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
    metrics = {"test_accuracy": float(accuracy_score(y_true, y_pred)), "precision": report["macro avg"]["precision"], "recall": report["macro avg"]["recall"], "f1": report["macro avg"]["f1-score"], "macro_f1": report["macro avg"]["f1-score"], "weighted_f1": report["weighted avg"]["f1-score"], "test_images": len(y_true), "averaging": "macro for precision/recall/f1"}
    return metrics, report, confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))


def save_evaluation(model_name, classes, y_true, probabilities, history, output, provenance):
    import pandas as pd
    import seaborn as sns
    plt = pyplot()
    output = Path(output)
    metrics_dir, figures_dir = output / "metrics", output / "figures" / model_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics, report, matrix = compute_metrics(y_true, probabilities, classes)
    metrics.update({"model": model_name, **provenance})
    write_json(metrics_dir / f"{model_name}.json", metrics)
    write_json(metrics_dir / f"{model_name}_history.json", history)
    pd.DataFrame([metrics]).to_csv(metrics_dir / f"{model_name}_metrics.csv", index=False)
    write_json(metrics_dir / f"{model_name}_classification_report.json", report)
    per_class = pd.DataFrame([{ "class": label, **report[label]} for label in classes])
    per_class.to_csv(metrics_dir / f"{model_name}_per_class.csv", index=False)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(metrics_dir / f"{model_name}_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(max(10, len(classes) * .38), max(8, len(classes) * .32)))
    sns.heatmap(matrix, xticklabels=classes, yticklabels=classes, cmap="Greens", ax=ax, annot=len(classes) <= 10, fmt="d")
    ax.set(xlabel="Predicted class", ylabel="Actual class", title="Held-out test confusion matrix")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, max(5, len(classes) * .25)))
    ax.barh(classes, per_class["recall"], color="#26734d")
    ax.set(xlim=(0, 1), xlabel="Recall", title="Per-class recall on held-out test set")
    fig.tight_layout()
    fig.savefig(figures_dir / "per_class_recall.png", dpi=160)
    plt.close(fig)
    plot_history(history, figures_dir)
    comparison = []
    for name in ("cnn", "efficientnet"):
        file = metrics_dir / f"{name}.json"
        if file.exists():
            entry = read_json(file)
            if entry.get("manifest_sha256") == provenance["manifest_sha256"]:
                comparison.append(entry)
    if len(comparison) == 2:
        frame = pd.DataFrame(comparison)[["model", "test_accuracy", "macro_f1", "weighted_f1"]]
        frame.to_csv(metrics_dir / "comparison.csv", index=False)
        ax = frame.set_index("model").plot.bar(ylim=(0, 1), color=["#26734d", "#89af79", "#c5dabb"], rot=0, figsize=(8, 5))
        ax.set_ylabel("Score")
        ax.set_title("Independent models on the same held-out test manifest")
        ax.figure.tight_layout()
        ax.figure.savefig(output / "figures/comparison.png", dpi=160)
        plt.close(ax.figure)
    else:
        for stale in (metrics_dir / "comparison.csv", output / "figures/comparison.png"):
            stale.unlink(missing_ok=True)
    return metrics
