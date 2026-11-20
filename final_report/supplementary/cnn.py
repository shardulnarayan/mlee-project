"""Manifest-aligned CNN reruns; imports ML dependencies only when requested."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models

from .evidence import (ROOT, CLASSES, image_path, local_path, manifest, scores,
                       sha256, write_csv, write_json)
from src.train_dl import train_transforms, val_test_transforms


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Some backends do not implement every deterministic operation.
    torch.use_deterministic_algorithms(True, warn_only=True)


def prepare_label_free(rows, output):
    """Apply segmentation to every class, without using its label to decide."""
    import cv2
    from src.dip.pipeline import process_image
    destination = Path(output) / "label_free_images"
    for index, row in enumerate(rows, 1):
        target = local_path(destination / row["split"] / row["class"] / (row["filename"] + ".png"))
        # Always regenerate: silently reusing a stale mask cache changes provenance.
        bgr = cv2.imdecode(np.fromfile(image_path(row, "raw"), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Unreadable raw image: {row}")
        cv2.setRNGSeed(0)
        result = process_image(bgr, segment=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".png", result["final"])
        if not ok:
            raise ValueError(f"Could not encode {target}")
        encoded.tofile(target)
        if index % 100 == 0:
            print(f"Label-free preprocessing: {index}/{len(rows)}", flush=True)


class ManifestImages(Dataset):
    def __init__(self, rows, variant, output, transform=val_test_transforms):
        self.rows, self.variant, self.output, self.transform = rows, variant, Path(output), transform

    def path(self, row):
        if self.variant == "label_free":
            return local_path(self.output / "label_free_images" / row["split"] / row["class"] /
                              (row["filename"] + ".png"))
        return image_path(row, self.variant)

    def image(self, index):
        with Image.open(self.path(self.rows[index])) as handle:
            return handle.convert("RGB")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.transform(self.image(index)), CLASSES.index(row["class"])


def checkpoint_path(output, variant, seed):
    return Path(output) / "checkpoints" / f"{variant}_seed{seed}.pth"


def load_model(output, variant, seed, device):
    path = checkpoint_path(output, variant, seed)
    meta_path = path.with_suffix(".json")
    meta = json.loads(local_path(meta_path).read_text(encoding="utf-8"))
    if meta["classes"] != CLASSES or meta["manifest_sha256"] != sha256(ROOT / "data/split_manifest.csv"):
        raise ValueError("Checkpoint class order or manifest differs from this evaluation")
    if meta["variant"] != variant or meta["seed"] != seed or meta["checkpoint_sha256"] != sha256(path):
        raise ValueError("Checkpoint identity/hash does not match its metadata")
    model = models.mobilenet_v2(weights=None, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(local_path(path), map_location="cpu", weights_only=True))
    return model.to(device).eval(), meta


def evaluate(model, dataset, device, batch_size):
    truth, predicted = [], []
    with torch.inference_mode():
        for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            indices = model(images.to(device)).argmax(1).cpu().tolist()
            truth.extend(CLASSES[i] for i in labels.tolist())
            predicted.extend(CLASSES[i] for i in indices)
    summary, per_class = scores(truth, predicted)
    return summary, per_class, predicted


def train_runs(args):
    rows = manifest()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if "label_free" in args.variants:
        prepare_label_free(rows, output)
    recipe = dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=0.001,
                  selection="validation_accuracy_first_max", backbone="ImageNet MobileNetV2",
                  train_mode="whole model.train(): frozen backbone parameters, updating BN running state",
                  classes=CLASSES, manifest_sha256=sha256(ROOT / "data/split_manifest.csv"),
                  source_training_sha256=sha256(ROOT / "src/train_dl.py"),
                  source_dip_sha256=sha256(ROOT / "src/dip/pipeline.py"),
                  runner_sha256=sha256(__file__), torch_version=torch.__version__,
                  device=str(device), threads=args.threads, label_free_encoding="PNG, regenerated from raw, cv2 RNG seed 0")
    all_runs = []
    for variant in args.variants:
        for seed in args.seeds:
            save = checkpoint_path(output, variant, seed)
            meta_path = save.with_suffix(".json")
            if save.exists() or meta_path.exists():
                if not args.resume:
                    raise FileExistsError(f"Refusing to overwrite {save}; choose a new output or use --resume")
                _, meta = load_model(output, variant, seed, device)
                if meta["recipe"] != recipe:
                    raise ValueError("Cannot resume a checkpoint trained with a different recipe/environment")
                all_runs.append(meta["metrics"])
                print(f"Verified existing {variant}, seed {seed}", flush=True)
                continue
            seed_all(seed)
            train_rows = [r for r in rows if r["split"] == "train"]
            val_rows = [r for r in rows if r["split"] == "val"]
            test_rows = [r for r in rows if r["split"] == "test"]
            train = ManifestImages(train_rows, variant, output, train_transforms)
            val = ManifestImages(val_rows, variant, output)
            test = ManifestImages(test_rows, variant, output)
            labels = [CLASSES.index(r["class"]) for r in train_rows]
            counts = np.bincount(labels, minlength=len(CLASSES))
            if (counts == 0).any():
                raise ValueError("Each training class must have at least one image")
            sampler = WeightedRandomSampler(torch.tensor([1 / counts[i] for i in labels], dtype=torch.double),
                                            num_samples=len(train), replacement=True)
            loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=0)
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
            model = model.to(device)
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
            loss_fn = nn.CrossEntropyLoss()
            best, best_epoch, best_state = -1, None, None
            epoch_rows = []
            for epoch in range(1, args.epochs + 1):
                print(f"{variant}, seed {seed}, epoch {epoch}/{args.epochs}", flush=True)
                model.train()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                for images, targets in loader:
                    optimizer.zero_grad()
                    loss = loss_fn(model(images.to(device)), targets.to(device))
                    loss.backward()
                    optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                model.eval()
                val_metrics, _, _ = evaluate(model, val, device, args.batch_size)
                if val_metrics["accuracy"] > best:
                    best, best_epoch = val_metrics["accuracy"], epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epoch_rows.append(dict(variant=variant, seed=seed, epoch=epoch,
                                       train_seconds=elapsed, validation_accuracy=val_metrics["accuracy"]))
            model.load_state_dict(best_state)
            model.eval()
            metrics, per_class, predicted = evaluate(model, test, device, args.batch_size)
            result = dict(variant=variant, seed=seed, best_epoch=best_epoch, best_val_accuracy=best,
                          mean_train_seconds=float(np.mean([r["train_seconds"] for r in epoch_rows])), **metrics)
            save.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, local_path(save))
            write_csv(output / f"{variant}_seed{seed}_epochs.csv", epoch_rows)
            write_csv(output / f"{variant}_seed{seed}_per_class.csv", per_class)
            write_csv(output / f"{variant}_seed{seed}_predictions.csv", [
                dict(filename=r["filename"], group=r["group"], true_class=r["class"], predicted_class=y)
                for r, y in zip(test_rows, predicted)])
            write_json(meta_path, dict(variant=variant, seed=seed, classes=CLASSES, recipe=recipe,
                       manifest_sha256=recipe["manifest_sha256"], checkpoint_sha256=sha256(save), metrics=result))
            all_runs.append(result)
            write_csv(output / "seeded_runs.csv", all_runs)
            print(result, flush=True)
    write_csv(output / "seeded_runs.csv", all_runs)
