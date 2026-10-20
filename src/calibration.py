"""Temperature scaling and embedding-distance rejection, independent of test data."""
import numpy as np


def temperature_scale(probabilities, temperature=1.):
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    logits = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=-1, keepdims=True)


def fit_temperature(probabilities, labels):
    from scipy.optimize import minimize_scalar
    labels = np.asarray(labels, dtype=int)
    def nll(log_temperature):
        calibrated = temperature_scale(probabilities, np.exp(log_temperature))
        return float(-np.log(np.clip(calibrated[np.arange(len(labels)), labels], 1e-12, 1)).mean())
    result = minimize_scalar(nll, bounds=(-3., 3.), method="bounded")
    if not result.success:
        raise ValueError("Temperature optimization failed.")
    return float(np.exp(result.x))


def normalize_embeddings(embeddings):
    array = np.asarray(embeddings, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


def embedding_distances(embeddings, predicted, prototypes):
    unit = normalize_embeddings(embeddings)
    return 1 - np.sum(unit * prototypes[np.asarray(predicted, dtype=int)], axis=1)


def ece(probabilities, labels, bins=15):
    probabilities = np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    error = 0.
    for i in range(bins):
        mask = (confidence >= i / bins) & (confidence < (i + 1) / bins if i < bins - 1 else confidence <= 1.)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def rejection_statistics(probabilities, distances, labels, threshold, distance_threshold):
    accepted = (probabilities.max(axis=1) >= threshold) & (distances <= distance_threshold)
    return {"images": len(accepted), "accepted": int(accepted.sum()), "coverage": float(accepted.mean()), "accepted_accuracy": float((probabilities.argmax(axis=1)[accepted] == np.asarray(labels)[accepted]).mean()) if labels is not None and accepted.any() else None}


def choose_threshold(id_probs, id_distances, labels, ood_probs, ood_distances, target_accuracy=.9, max_ood_acceptance=.05, ood_groups=None):
    best = None
    group_masks = [np.asarray(ood_groups) == value for value in np.unique(ood_groups)] if ood_groups is not None else [np.ones(len(ood_probs), dtype=bool)]
    for threshold in np.r_[np.linspace(.35, .99, 33), .995, .999]:
        for cutoff in np.unique(np.quantile(id_distances, [.1, .2, .3, .4, .5, .65, .8, .9, .95, .98, 1.])):
            known = rejection_statistics(id_probs, id_distances, labels, threshold, cutoff)
            unknown = rejection_statistics(ood_probs, ood_distances, None, threshold, cutoff)
            group_acceptance = [rejection_statistics(ood_probs[mask], ood_distances[mask], None, threshold, cutoff)["coverage"] for mask in group_masks]
            if known["accepted_accuracy"] is not None and known["accepted_accuracy"] >= target_accuracy and unknown["coverage"] <= max_ood_acceptance and max(group_acceptance) <= max_ood_acceptance:
                if best is None or known["coverage"] > best["coverage"]:
                    best = {"threshold": float(threshold), "distance_threshold": float(cutoff), "coverage": known["coverage"]}
    return best
