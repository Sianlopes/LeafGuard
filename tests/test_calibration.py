import numpy as np
from calibrate import partition_groups
from src.calibration import ece, rejection_statistics, choose_threshold


def test_calibration_partition_keeps_same_leaf_together():
    rows = [{"path": f"{leaf}-{view}", "group_id": str(leaf)} for leaf in range(12) for view in range(3)]
    partitions = partition_groups(rows, 42)
    assert sum(map(len, partitions)) == len(rows)
    sets = [{row["group_id"] for row in part} for part in partitions]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


def test_ood_rule_uses_embedding_and_reports_coverage():
    known = np.array([[.95, .05], [.05, .95]] * 10)
    ood = np.array([[.99, .01]] * 20)  # High-confidence unknown images.
    chosen = choose_threshold(known, np.full(20, .1), np.array([0, 1] * 10), ood, np.full(20, .8))
    assert chosen is not None
    statistics = rejection_statistics(ood, np.full(20, .8), None, chosen["threshold"], chosen["distance_threshold"])
    assert statistics["accepted"] == 0 and statistics["accepted_accuracy"] is None
    assert ece(known, np.array([0, 1] * 10)) < .051


def test_easy_objects_cannot_mask_unsupported_leaf_acceptance():
    known = np.array([[.95, .05]] * 40)
    unknown = np.array([[.99, .01]] * 100)
    distances = np.array([.1] + [.8] * 99)
    arguments = (known, np.full(40, .1), np.zeros(40, dtype=int), unknown, distances)
    assert choose_threshold(*arguments) is not None
    groups = np.array(["unsupported_leaves"] + ["unrelated_objects"] * 99)
    assert choose_threshold(*arguments, ood_groups=groups) is None
