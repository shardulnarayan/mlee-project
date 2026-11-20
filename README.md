# Cucumber Leaf Pest and Disease Classification

A five-class image classification study comparing digital image preprocessing, handcrafted features with SVM and Random Forest, and MobileNetV2 transfer learning. Labels: **Whitefly, Red_Mite, Yellow_Mosaic, Leaf_Miner and Healthy**. Red_Mite and Yellow_Mosaic use tea and luffa proxy datasets, respectively.

## Approach

1. Assemble source images, deduplicate and apply the committed source-group split.
2. Produce 224 x 224 inputs with adaptive foreground masking and CLAHE; use Otsu and Canny for diagnostic previews.
3. Train SVM/Random Forest on HOG, LBP and HSV features, and MobileNetV2 on raw and processed images.
4. Compare the shared holdout, feature ablations, embedding SVM, minority-class learning, grouped cross-validation, Grad-CAM, synthetic noise and storage.

## Main results

The shared dataset contains **1,004 images: 702 training, 151 validation and 151 test**. Crops of each original pest photograph stay in one split.

| Experiment | Test accuracy | Test macro-F1 |
|---|---:|---:|
| SVM, augmented DIP features | 88.74% | 0.8017 |
| Random Forest, augmented DIP features | 93.38% | 0.8611 |
| MobileNetV2, DIP images | 89.40% | 0.8588 |
| MobileNetV2, raw images | 95.36% | Approximately 0.93 (roadmap precision) |
| Random Forest, LBP + colour selected on validation | 96.69% | 0.9094 |
| MobileNetV2 embeddings + SVM | 95.36% | 0.8774 |

Five-fold source-group CV of the fixed LBP + colour Random Forest yielded macro-F1 **0.8529 +/- 0.0660**. Matched image storage fell **80.40%**; INT8 weight storage reduced checkpoint size **70.98%**. Quantized weights were restored to FP32 for evaluation; native INT8 execution was not measured.

Raw MobileNetV2 led on clean images. In the tracked whole-input Gaussian-noise run, accuracy was 34.44% raw and 55.63% DIP. This is synthetic corruption; independent field performance remains unmeasured. [Complete results and trade-offs](docs/RESULTS.md).

## Getting started

Annex C reproducibility code is in
[`final_report/supplementary/`](final_report/supplementary/README.md).
It provides seeded CNN training, foreground-attention analysis and synthetic
camera-fault evaluation. Execution status is recorded in `table_status.json`.
See the [Annex C methods and evidence guide](final_report/ANNEX_C_REPLACEMENT.md).

Run from the repository root in PowerShell:

```powershell
python -m venv mlee-env
.\mlee-env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Obtain the source dataset and place it under `data/raw/`. Images and trained weights are excluded from Git. Preserve the committed manifest, then run:

```powershell
python -m src.data_prep.split_dataset --dry-run
python -m src.data_prep.split_dataset
python -m src.dip.pipeline
python -m src.data_prep.split_dataset
python -m src.classical_ml.train --no-cache
python -m src.train_dl
python -m src.evaluate_dl
```

[Reproducing the experiments](docs/REPRODUCING.md) lists all commands, checkpoints, output paths and evidence limitations.

## Repository guide

| Path | Contents |
|---|---|
| `src/` | Dataset preparation, DIP, models, evaluation and analysis code |
| `data/split_manifest.csv` | Canonical image, label, source-group and split assignments |
| `results/metrics/` | Predictions, summaries, ablations, CV and storage tables |
| `results/figures/` | Final and failed DIP iterations, confusion matrices, Grad-CAM and analysis figures |
| `results/storage_reduction/` | Checkpoint storage comparison |
| `models/` | Local trained weights; Git retains a placeholder |
| [Methodology](docs/METHODOLOGY.md) | Dataset, algorithm explanations and evaluation rationale |
| [Results](docs/RESULTS.md) | Full experimental results and discussion |
| [Reproducing](docs/REPRODUCING.md) | Setup, commands, output inventory and provenance |
| `docs/references/` | Project roadmap, dataset sources and pipeline design documents |

## Interpretation and limitations

Healthy has six test images and Leaf_Miner ten. Group CV measures split sensitivity for previously selected configurations. Proxy plants and source-specific appearance limit field claims. The DIP code uses known class folders to skip segmentation for tight pest crops; deployment would need a label-independent decision. Grad-CAM examples are illustrative. Historical CNN epoch times lack a tracked benchmark log, and some raw CNN results remain available only at roadmap precision.

## Contributors

- Ayushman
- Shardul
- Paarth Juneja
