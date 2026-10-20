"""Read only public source METADATA, without downloading the 2 GB image archive."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import Counter, defaultdict
from src.common import ROOT, read_json, write_json
from src.dataset import fetch, leaf_identifier, DEFAULT_REVISION, normalized_leaf_map


def main():
    base = f"https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{DEFAULT_REVISION}"
    directory = ROOT / "data/raw/source_audit"
    leaf_map = normalized_leaf_map(read_json(fetch(base + "/leaf_grouping/leaf-map.json", directory / "leaf-map.json")))
    classes, leaf_splits, failures = Counter(), defaultdict(set), []
    counts = Counter()
    for split in ("train", "test"):
        listing = fetch(base + f"/splits/color_{split}.txt", directory / f"color_{split}.txt")
        for relative in listing.read_text(encoding="utf-8").splitlines():
            if not relative.strip():
                continue
            path = Path(relative)
            label = path.parent.name
            counts[split] += 1
            classes[label] += 1
            try:
                leaf = leaf_identifier(path.name, label, leaf_map)
                leaf_splits[leaf].add(split)
            except ValueError as exc:
                failures.append(str(exc))
    audit = {"revision": DEFAULT_REVISION, "source_counts": dict(counts), "source_classes": len(classes), "unresolved_leaf_ids": len(failures), "examples": failures[:10], "cross_split_leaf_groups": sum(len(s) > 1 for s in leaf_splits.values()), "scope": "Public metadata only; no image decode, training or accuracy evaluation."}
    write_json(ROOT / "reports/source_audit.json", audit)
    print(audit)


if __name__ == "__main__":
    main()
