import csv
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from src.dataset import prepare_dataset, write_rows, load_manifest, leaf_identifier, HashTree, normalized_leaf_map
from src.common import sha256, write_json


@pytest.fixture
def local_data(tmp_path):
    root = tmp_path / "raw"
    rows = []
    rng = np.random.default_rng(42)
    for label in ("Apple___healthy", "Potato___Early_blight"):
        for group in range(12):
            for view in range(2):
                relative = f"{label}/{group}_{view}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
                rows.append({"path": relative, "label": label, "leaf_id": f"{label}-{group}", "split": "test" if group >= 10 else "train"})
    write_rows(root / "groups.csv", rows)
    return root, rows


def test_split_groups_reproducible_and_official_test_preserved(local_data, tmp_path):
    root, source = local_data
    first = prepare_dataset(root, tmp_path / "one")
    second = prepare_dataset(root, tmp_path / "two")
    assert [(r["path"], r["split"]) for r in first] == [(r["path"], r["split"]) for r in second]
    for key in ("leaf_id", "pixel_sha256", "group_id"):
        for value in {r[key] for r in first}:
            assert len({r["split"] for r in first if r[key] == value}) == 1
    assert all(r["split"] == "test" for r in first if r["source_split"] == "test")
    assert {r["split"] for r in first} == {"train", "validation", "calibration", "test"}


def test_corruption_is_reported(local_data, tmp_path):
    root, source = local_data
    bad = root / source[0]["path"]
    bad.write_bytes(b"broken")
    output = tmp_path / "processed"
    rows = prepare_dataset(root, output)
    assert len(rows) == len(source) - 1
    assert str(bad) in {r["path"] for r in json.loads((output / "excluded.json").read_text())}


def test_cross_split_duplicates_quarantined(local_data, tmp_path):
    root, source = local_data
    train, test = source[0], source[20]
    (root / test["path"]).write_bytes((root / train["path"]).read_bytes())
    rows = prepare_dataset(root, tmp_path / "out")
    assert str(root / train["path"]) not in {r["path"] for r in rows}
    assert str(root / test["path"]) in {r["path"] for r in rows}


def test_missing_groups_refused(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    with pytest.raises(ValueError, match="Physical leaf IDs"):
        prepare_dataset(root, tmp_path / "out")


def test_manifest_tampering_detected(local_data, tmp_path):
    root, _ = local_data
    output = tmp_path / "processed"
    prepare_dataset(root, output)
    with (output / "manifest.csv").open("a") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="Manifest changed"):
        load_manifest(output)


def test_group_lookup_uses_source_metadata():
    mapping = {"rs_early.b 7557": ["Potato___Early_blight:::115.0"]}
    assert leaf_identifier("uuid___RS_Early.B 7557.JPG", "Potato___Early_blight", mapping) == "Potato___Early_blight:::115.0"
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        leaf_identifier("uuid___not-known.JPG", "Potato___Early_blight", mapping)


def test_genuine_source_aliases():
    mapping = normalized_leaf_map({"com.g_fl 9730": ["Tomato___Target_Spot:::12.0"], "jr_frge.s 3094": ["Apple_Frogeye Spot:::31.0"]})
    assert leaf_identifier("uuid___Com.G_TgS_FL 9730.JPG", "Tomato___Target_Spot", mapping) == "Tomato___Target_Spot:::12.0"
    assert leaf_identifier("uuid___JR_FrgE.S 3094.JPG", "Apple___Black_rot", mapping) == "Apple___Black_rot:::31.0"


def test_hash_tree_candidates():
    tree = HashTree()
    for value, index in ((0, 0), (1, 1), (7, 2), (255, 3)):
        tree.add(value, index)
    assert set(tree.query(0, 2)) == {0, 1}


def test_publisher_policy_is_explicit_and_keeps_unknown_ids_unknown(local_data, tmp_path):
    root, source = local_data
    for row in source:
        row["source_group"] = row["leaf_id"]
        row["group_basis"] = "physical_leaf_metadata"
    for row in source[:2]:
        row["leaf_id"] = ""
        row["source_group"] = "source-image:Apple___healthy:0"
        row["group_basis"] = "source_image_unverified"
    write_rows(root / "groups.csv", source)
    write_json(root / "provenance.json", {"repository": "mohanty/PlantVillage", "group_policy": "publisher", "groups_csv_sha256": sha256(root / "groups.csv")})
    with pytest.raises(ValueError, match="unverified leaf ID"):
        prepare_dataset(root, tmp_path / "strict")
    rows = prepare_dataset(root, tmp_path / "publisher", group_policy="publisher")
    unknown = [row for row in rows if row["group_basis"] == "source_image_unverified"]
    assert len(unknown) == 2 and all(row["leaf_id"] == "" for row in unknown)
    metadata = json.loads((tmp_path / "publisher/metadata.json").read_text())
    assert not metadata["complete_physical_leaf_verification"]
    assert metadata["unverified_physical_leaf_records"] == 2


def test_near_duplicates_purge_training_match_preserve_test(local_data, tmp_path):
    root, source = local_data
    train_path = root / source[0]["path"]
    test_path = root / source[20]["path"]
    pixels = np.asarray(Image.open(train_path)).astype(np.int16)
    Image.fromarray(np.clip(pixels + 1, 0, 255).astype(np.uint8)).save(test_path)
    rows = prepare_dataset(root, tmp_path / "near")
    assert str(train_path) not in {r["path"] for r in rows}
    assert str(test_path) in {r["path"] for r in rows}
    assert (tmp_path / "near/near_duplicates.csv").exists()


def test_no_predefined_splits(local_data, tmp_path):
    root, rows = local_data
    for row in rows:
        row["split"] = ""
    write_rows(root / "groups.csv", rows)
    result = prepare_dataset(root, tmp_path / "allnew")
    assert {r["split"] for r in result} == {"train", "validation", "calibration", "test"}


@pytest.mark.integration
def test_training_and_evaluation_entry_points(local_data, tmp_path, monkeypatch):
    pytest.importorskip("tensorflow")
    import sys
    import train
    import evaluate
    root, _ = local_data
    processed = tmp_path / "processed"
    out = tmp_path / "model"
    reports = tmp_path / "reports"
    prepare_dataset(root, processed)
    monkeypatch.setattr(sys, "argv", ["train.py", "--model", "cnn", "--data-dir", str(processed), "--output", str(out), "--reports-dir", str(reports), "--epochs", "1", "--batch-size", "8", "--image-size", "32", "--cpu"])
    train.main()
    assert (out / "best.keras").exists()
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--model-dir", str(out), "--data-dir", str(processed), "--output", str(reports), "--cpu"])
    evaluate.main()
    metrics = json.loads((reports / "metrics/cnn.json").read_text())
    assert metrics["test_images"] == 8 and 0 <= metrics["test_accuracy"] <= 1
    assert metrics["manifest_sha256"] == sha256(processed / "manifest.csv")
