"""Five-fold source-group cross-validation for the classical DIP models."""

import csv
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from src import config
from src.classical_ml.augment import augment_image
from src.classical_ml.feature_ablation import feature_columns
from src.classical_ml.features import extract_features, load_image
from src.classical_ml.train import build_rf, build_svm


def augment_fold(paths, labels, rng):
    """Balance the fold's training set using only its own originals."""
    extras, targets = [], []
    counts = np.bincount(labels, minlength=len(config.CLASSES))
    for label, count in enumerate(counts):
        missing = counts.max() - count
        if missing:
            originals = [load_image(path) for path in paths[labels == label]]
            for index in range(missing):
                extras.append(extract_features(augment_image(originals[index % count], rng)))
                targets.append(label)
    return np.asarray(extras), np.asarray(targets, dtype=int)


def main():
    root = config.PROCESSED_DIR
    if os.name == "nt" and root.drive and not str(root).startswith("\\\\"):
        root = Path("\\\\?\\" + str(root))
    with (config.ROOT_DIR / "data" / "split_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle))
    paths = np.array([root / row["split"] / row["class"] / row["filename"]
                      for row in manifest], dtype=object)
    labels = np.array([config.CLASS_TO_IDX[row["class"]] for row in manifest])
    groups = np.array([row["group"] for row in manifest])
    if len(set(paths)) != len(paths) or any(not group.strip() for group in groups):
        raise ValueError("Manifest has duplicate paths or missing groups")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    splits = list(splitter.split(paths, labels, groups))
    assigned = np.concatenate([held for _, held in splits])
    classes = np.arange(len(config.CLASSES))
    if not np.array_equal(np.sort(assigned), np.arange(len(paths))):
        raise ValueError("Each image must be held out exactly once")
    for train, held in splits:
        if set(groups[train]) & set(groups[held]):
            raise ValueError("A source group crosses a fold boundary")
        if not (np.array_equal(np.unique(labels[train]), classes)
                and np.array_equal(np.unique(labels[held]), classes)):
            raise ValueError("Each side of every fold must contain all five classes")

    print(f"Extracting features for {len(paths)} manifest images", flush=True)
    features = np.stack([extract_features(load_image(path)) for path in paths])
    blocks = feature_columns()
    reduced = np.concatenate([blocks["LBP"], blocks["Colour"]])
    scores, membership = [], []
    with threadpool_limits(limits=4):
        for fold, (train, held) in enumerate(splits, 1):
            print(f"Fold {fold}: {len(train)} train, {len(held)} held out", flush=True)
            for index in held:
                membership.append({"fold": fold, **manifest[index]})
            rng = np.random.default_rng(config.SEED + fold)
            extra_x, extra_y = augment_fold(paths[train], labels[train], rng)
            train_x = np.concatenate((features[train], extra_x)) if len(extra_y) else features[train]
            train_y = np.concatenate((labels[train], extra_y))
            for name, model, cols in (
                ("SVM full", build_svm(C=10), slice(None)),
                ("RF full", build_rf(max_depth=None), slice(None)),
                ("RF LBP + Colour", build_rf(max_depth=None), reduced),
            ):
                model.fit(train_x[:, cols], train_y)
                predictions = model.predict(features[held][:, cols])
                precision, recall, f1, support = precision_recall_fscore_support(
                    labels[held], predictions, labels=classes, zero_division=0,
                )
                row = {
                    "model": name, "fold": fold, "held_images": len(held),
                    "accuracy": accuracy_score(labels[held], predictions),
                    "macro_precision": precision.mean(), "macro_recall": recall.mean(),
                    "macro_f1": f1.mean(),
                }
                for index, label in enumerate(config.CLASSES):
                    row.update({f"{label}_{metric}": values[index] for metric, values in
                                (("precision", precision), ("recall", recall),
                                 ("f1", f1), ("support", support))})
                scores.append(row)
                print(f"  {name}: macro-F1 {row['macro_f1']:.4f}", flush=True)

    summary = []
    for name in dict.fromkeys(row["model"] for row in scores):
        model_rows = [row for row in scores if row["model"] == name]
        for metric in model_rows[0]:
            if metric in ("model", "fold"):
                continue
            values = np.array([row[metric] for row in model_rows])
            summary.append({"model": name, "metric": metric, "mean": values.mean(),
                            "sd": values.std(ddof=1)})

    for filename, rows in (
        ("group_cv_folds.csv", scores),
        ("group_cv_summary.csv", summary),
        ("group_cv_membership.csv", membership),
    ):
        output = config.RESULTS_DIR / "metrics" / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
