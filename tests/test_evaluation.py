import numpy as np
from src.evaluation import compute_metrics, save_evaluation


def test_metrics_include_missing_predictions():
    metrics, report, matrix = compute_metrics([0, 1, 2], np.array([[.9, .05, .05], [.8, .1, .1], [.7, .2, .1]]), ["a", "b", "c"])
    assert metrics["test_accuracy"] == 1 / 3
    assert report["c"]["recall"] == 0
    assert matrix.shape == (3, 3)


def test_real_evaluation_outputs_and_comparison(tmp_path):
    classes = ["Apple___healthy", "Potato___Early_blight"]
    probabilities = np.array([[.8, .2], [.1, .9]])
    history = {"accuracy": [.5, .7], "val_accuracy": [.5, .6], "loss": [.7, .5], "val_loss": [.8, .6]}
    for name in ("cnn", "efficientnet"):
        save_evaluation(name, classes, [0, 1], probabilities, history, tmp_path, {"manifest_sha256": "fixture", "model_sha256": "fixture"})
    assert (tmp_path / "metrics/comparison.csv").exists()
    assert (tmp_path / "figures/comparison.png").exists()
    assert (tmp_path / "figures/cnn/confusion_matrix.png").exists()
