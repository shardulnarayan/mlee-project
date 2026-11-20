"""Dependency-free evidence checks and report-table assembly."""

import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSES = sorted(["Whitefly", "Red_Mite", "Yellow_Mosaic", "Leaf_Miner", "Healthy"])
VARIANTS = ("raw", "dip", "label_free")


def local_path(path):
    path = Path(path).resolve()
    if os.name == "nt" and not str(path).startswith("\\\\"):
        return Path("\\\\?\\" + str(path))
    return path


def read_csv(path):
    with local_path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No observations to write: {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with local_path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    local_path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with local_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest():
    rows = read_csv(ROOT / "data/split_manifest.csv")
    seen, groups, counts = set(), {}, {}
    for row in rows:
        if row["class"] not in CLASSES or row["split"] not in ("train", "val", "test"):
            raise ValueError(f"Unexpected manifest label/split: {row}")
        filename = Path(row["filename"])
        if filename.name != row["filename"] or row["filename"] in (".", ".."):
            raise ValueError("Manifest filename must be one path component")
        key = (row["split"], row["class"], row["filename"])
        if key in seen or not row["group"].strip():
            raise ValueError("Duplicate manifest path or empty source group")
        seen.add(key)
        previous = groups.setdefault(row["group"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"Source group crosses splits: {row['group']}")
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    if counts != {"train": 702, "val": 151, "test": 151}:
        raise ValueError(f"Expected the canonical 1,004-image manifest, got {counts}")
    return rows


def image_path(row, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    kind = "processed" if variant == "dip" else "raw"
    return local_path(ROOT / "data" / kind / row["split"] / row["class"] / row["filename"])


def scores(truth, predictions):
    """Five-class metrics, including absent classes as zero (same denominator)."""
    if not truth or len(truth) != len(predictions):
        raise ValueError("Expected equal nonempty truth and prediction lists")
    if any(label not in CLASSES for label in list(truth) + list(predictions)):
        raise ValueError("Unknown class name")
    per_class = []
    for label in CLASSES:
        tp = sum(a == b == label for a, b in zip(truth, predictions))
        support = truth.count(label)
        predicted = predictions.count(label)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(dict(class_name=label, precision=precision, recall=recall,
                              f1=f1, support=support))
    summary = dict(test_images=len(truth), accuracy=sum(a == b for a, b in zip(truth, predictions)) / len(truth),
                   macro_precision=statistics.mean(r["precision"] for r in per_class),
                   macro_recall=statistics.mean(r["recall"] for r in per_class),
                   macro_f1=statistics.mean(r["f1"] for r in per_class))
    return summary, per_class


def mean_sd(values):
    values = [float(x) for x in values]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Need finite observations")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def check_inputs(variants):
    rows = manifest()
    missing = []
    for variant in variants:
        for row in rows:
            path = image_path(row, variant)
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    return {"manifest_images": len(rows), "variants": list(variants),
            "source_group_leakage": False, "missing_image_count": len(missing),
            "first_missing_images": missing[:10],
            "manifest_sha256": sha256(ROOT / "data/split_manifest.csv")}


def assemble_tables(output):
    """Assemble tables from measurement files and record pending stages."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    status = []
    seed_csv = output / "seeded_runs.csv"
    if seed_csv.exists():
        rows = read_csv(seed_csv)
        summary = []
        for variant in sorted({r["variant"] for r in rows}):
            selected = [r for r in rows if r["variant"] == variant]
            if len({r["seed"] for r in selected}) != len(selected):
                raise ValueError("Duplicate seed/variant measurements")
            for metric in ("best_val_accuracy", "accuracy", "macro_f1", "mean_train_seconds"):
                mean, sd = mean_sd(r[metric] for r in selected)
                summary.append(dict(variant=variant, metric=metric, runs=len(selected), mean=mean,
                                    sample_sd="" if sd is None else sd))
        write_csv(output / "table6_seeded_summary.csv", summary)
        status.append("Table 6: seeded-run summary written")
    else:
        status.append("Table 6: pending seeded training")
    attention = output / "attention_per_image.csv"
    if attention.exists():
        rows = read_csv(attention)
        summary = []
        for variant in sorted({r["variant"] for r in rows}):
            for label in CLASSES + ["ALL"]:
                subset = [r for r in rows if r["variant"] == variant and
                          (label == "ALL" or r["class_name"] == label)]
                usable = [r for r in subset if r["status"] == "ok"]
                if not subset:
                    continue
                item = dict(variant=variant, class_name=label, eligible_records=len(subset),
                            usable_records=len(usable), excluded_records=len(subset)-len(usable))
                for metric in ("attention_share", "mask_area"):
                    vals = [r[metric] for r in usable]
                    mean, sd = mean_sd(vals) if vals else (None, None)
                    item[metric + "_mean"] = "" if mean is None else mean
                    item[metric + "_sample_sd"] = "" if sd is None else sd
                summary.append(item)
        write_csv(output / "table7_attention_summary.csv", summary)
        status.append("Table 7: attention summary written (record-level pooling)")
    else:
        status.append("Table 7: pending attention evaluation")
    per_class = ROOT / "results/metrics/pj_holdout_per_class.csv"
    if per_class.exists():
        write_csv(output / "table8_existing_holdout_per_class.csv", read_csv(per_class))
        status.append("Table 8: existing team holdout class metrics copied with model labels")
    ablation = ROOT / "results/metrics/rf_feature_ablation.csv"
    cv = ROOT / "results/metrics/group_cv_summary.csv"
    if ablation.exists() and cv.exists():
        write_csv(output / "table9_existing_feature_ablation.csv", read_csv(ablation))
        write_csv(output / "table9_existing_group_cv.csv", read_csv(cv))
        status.append("Table 9: existing feature-ablation and grouped-CV evidence copied")
    write_json(output / "table_status.json", {"status": status})
    return status
