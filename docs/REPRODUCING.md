# Reproducing the experiments

Run commands from the repository root. Source images, feature caches and trained weights are local artifacts excluded from Git. Predictions, metrics and figures listed below are tracked.

## Setup and commands

From the repo root on Windows PowerShell:

```powershell
python -m venv mlee-env
.\mlee-env\Scripts\Activate.ps1
pip install -r requirements.txt
```

The images are not in Git. Put the raw dataset in `data/raw/` (or rebuild it from
`external_data/` with `crop_pests.py` and `organize_datasets.py`), then run in order:

```powershell
python -m src.data_prep.split_dataset            # apply the shared split to data/raw
python -m src.dip.pipeline                       # build data/processed
python -m src.data_prep.split_dataset            # check processed matches the split
python -m src.classical_ml.train --no-cache      # SVM + RF (add --no-augment for the baseline)
python -m src.train_dl                           # MobileNetV2, processed
python -m src.train_dl_raw                       # MobileNetV2, raw (ablation)
python -m src.evaluate_dl
python -m src.evaluate_dl_raw
python -m src.gradcam
python -m src.gradcam_raw
python -m src.background_swap
python -m src.storage_metric
python -m src.classical_ml.feature_ablation
python -m src.classical_ml.mobilenet_svm
python -m src.classical_ml.leaf_miner_learning_curve
python -u -m src.classical_ml.plot_feature_tsne
python -u -m src.classical_ml.measure_quantized_storage
python -u -m src.classical_ml.group_cross_validation
```

Note: `config.CLASSES` is not alphabetical, but `torchvision.ImageFolder` sorts classes
alphabetically. Compare predictions by class name, not index.
## Options and operational details

| Command or option | Purpose |
|---|---|
| `python -m src.data_prep.split_dataset --dry-run` | Preview moves |
| `python -m src.data_prep.split_dataset --dataset raw` | Apply the manifest only to raw images |
| `python -m src.data_prep.split_dataset --rebuild` | Recreate assignments from pooled originals; changes the experimental split and requires downstream results to be reconsidered |
| `python -m src.classical_ml.train --no-augment` | No-augmentation control |
| `python -m src.classical_ml.train --dataset raw` | Raw classical run; label separately from the processed benchmark |
| `python -m src.classical_ml.train --no-cache` | Refresh features after images or splits change |
| `python -m src.classical_ml.features path/to/image.jpg` | Inspect a feature vector |
| `python -m src.classical_ml.augment` | Generate augmentation examples; the default example is the raw variant |

Shared settings live in `src/config.py`: image size 224, seed 42 and five labels. Feature caches live in `data/features/`. The normal benchmark uses the same processed images for classical ML and the processed CNN. Raw runs are separate ablations.

`src/downsample_data.py` remains as a historical helper. Use the manifest script for canonical downsampling; the old helper depended on OS file order.

`models/svm_model.pkl` and `models/rf_model.pkl` are augmented processed models. `svm_model_processed_noaug.pkl` and `rf_model_processed_noaug.pkl` are no-augmentation counterparts. The CNN checkpoints are `models/mobilenetv2_cucumber.pth` and `models/mobilenetv2_raw.pth`. Obtain the original checkpoint or retrain before checkpoint-dependent analyses. The raw checkpoint and exact raw per-image evaluation records remain unavailable in the checked files.


## Source and output map

| Path | What it does | Effect / output |
|---|---|---|
| `src/config.py` | Paths, `IMG_SIZE`, `SEED = 42`, 70/15/15, class list | Single source of truth |
| `src/crop_pests.py` | Crops Whitefly / Leaf_Miner YOLO boxes from `external_data/` | Classification-ready pest crops |
| `src/organize_datasets.py` | Copies Red_Mite, Yellow_Mosaic, Healthy, cap 1,500 | Raw 5-class folders |
| `src/downsample_data.py` | Old cap to 300 (OS-order dependent) | **Superseded** by `split_dataset.py`; do not run |
| `src/data_prep/split_dataset.py` | Applies (or `--rebuild`s) the manifest to raw and processed | Identical leakage-safe split |
| `data/split_manifest.csv` | 1,004 rows: split, class, filename, source group | The split itself |
| `src/dip/pipeline.py` | Iteration 3 DIP pipeline | `data/processed/`, evolution plots |
| `src/classical_ml/features.py` | HOG + LBP + colour (6,208) | Feature vectors |
| `src/classical_ml/augment.py` | Train-only augmentation of small classes | Balanced training set |
| `src/classical_ml/train.py` | SVM + RF, val selection, test scoring (`--no-augment`, `--no-cache`, `--dataset raw`) | `models/*.pkl`, predictions, summary JSON |
| `src/train_dl.py` / `train_dl_raw.py` | MobileNetV2 on processed / raw | `mobilenetv2_cucumber.pth` / `mobilenetv2_raw.pth` |
| `src/evaluate_dl.py` / `evaluate_dl_raw.py` | Test report, confusion matrix, per-image CSV | `dl_classification_report*.txt`, `dl_predictions*.csv`; processed predictions tracked, raw records unavailable |
| `src/gradcam.py` / `gradcam_raw.py` | Grad-CAM, one image per class | `results/figures/gradcam*/` |
| `src/background_swap.py` | Gaussian-noise robustness test | `ood_background_swap_results.csv` |
| `src/storage_metric.py` | Whole-folder sizes, model size | `storage_metrics.csv` |
| `src/classical_ml/feature_ablation.py` | RF on 7 feature subsets | `rf_feature_ablation.csv` |
| `src/classical_ml/mobilenet_svm.py` | CNN embeddings + SVM | `mobilenet_svm_comparison.csv` |
| `src/classical_ml/leaf_miner_learning_curve.py` | Recall vs Leaf_Miner source groups | `leaf_miner_learning_curve.*` |
| `src/classical_ml/plot_feature_tsne.py` | t-SNE of HOG vs CNN features | `hog_mobilenet_tsne.png` |
| `src/classical_ml/measure_quantized_storage.py` | Matched storage + INT8 checkpoint size | `mobilenetv2_quantization.csv` |
| `src/classical_ml/group_cross_validation.py` | 5-fold source-group CV | `group_cv_*.csv` |
| `results/figures/` | Evolution plots (+ failed iterations), augmentation examples, confusion matrices, Grad-CAM, t-SNE, learning curve | Report figures |
| `results/metrics/` | Split summary, classical summaries and predictions, ablation, CV, noise test, storage | Report tables |
| `results/storage_reduction/` | Checkpoint storage CSV | DIP report metric |
| `docs/references/roadmap- MLEE project.pdf`, `docs/references/roadmap-MLEE-project.pdf` | Original and updated roadmap (updated one adds DL, ablation, Grad-CAM notes) | Plan and report drafts |
| `docs/references/MLEE project pipeline.pdf` | Two-track design, role split, dependencies | Project plan |
| `data/`, `models/`, `notebooks/` | — | Images and weights (Git-ignored); unused notebooks folder | Local only |

---

## Output checklist

All paths below are under `results/` unless indicated otherwise. Use main report rows for headline evidence, and the audit files or supplementary plots in appendices.

| File or set | Contents | Placement |
| --- | --- | --- |
| `../data/split_manifest.csv` from the results directory, `metrics/dataset_split_summary.csv` | Per-image source groups and train/val/test assignments. Class counts, duplicate counts and downsampling counts. | Dataset table plus audit manifest |
| `figures/{Healthy,Leaf_Miner,Red_Mite,Whitefly,Yellow_Mosaic}_evolution.png` | Five final DIP stage panels, each with input, mask, enhanced output, Otsu and Canny. | DIP report |
| `figures/Iteration_1_HSV_Fail/`, `figures/Iteration_2_Otsu_Fail/` | Five panels per iteration, same class names as above, plus the iteration-1 notes. | DIP iteration discussion or appendix |
| `figures/augmentation_examples_processed.png` | Classical training transformation examples. | Methods |
| `metrics/classical_ml_processed_summary.json`, `metrics/classical_ml_processed_noaug_summary.json` | Hyperparameters, validation/test metrics, inference timing and per-class scores for augmented and no-augmentation SVM/RF. Label timing with the measurement scope. | Classical baseline and augmentation table |
| `metrics/{rf,svm}_processed_test_predictions.csv`, `metrics/{rf,svm}_processed_noaug_test_predictions.csv` | Four classical per-image prediction files. | Audit appendix |
| `metrics/dl_predictions.csv` | AP's processed MobileNetV2 per-image predictions recovered from the original team repo at `41a46ea`. | Audit appendix |
| `metrics/pj_holdout_comparison.csv`, `metrics/pj_holdout_per_class.csv`, `metrics/pj_holdout_confusion_counts.csv` | Aggregate scores, class precision, recall, F1 and support and all matrix cells for the five manifest-checked processed model configurations. Classical validation values use the matching JSON summaries. | Main benchmark and appendix |
| `figures/model_comparison_confusion_matrices.png` | Shared-scale processed test heatmaps for augmented SVM/RF and MobileNetV2. | Main ML figure |
| `metrics/pj_raw_cnn_roadmap_metrics.csv`, `metrics/pj_cnn_raw_vs_dip.csv` | AP's raw class table transcribed at roadmap precision, plus raw/DIP test metrics and approximate per-epoch times. | Raw-vs-DIP results |
| `figures/gradcam/`, `figures/gradcam_raw/` | Five processed and five raw Grad-CAM JPGs, one per class. | Show paired examples with caveats |
| `metrics/ood_background_swap_results.csv` | Clean and corrupted accuracy and drop for both CNNs. | Optional stress-test table |
| `metrics/rf_feature_ablation.csv` | Seven feature subsets, dimension, validation/test metrics and class recalls. | Ablation table |
| `metrics/mobilenet_svm_comparison.csv` | Matched handcrafted and embedding SVM plus AP head, including validation/test metrics and class recalls. | Hybrid table |
| `metrics/leaf_miner_learning_curve.csv`, `figures/leaf_miner_learning_curve.png` | Per-draw group count, crop count and validation outcomes, plus summarized plot. | Minority-class analysis |
| `figures/hog_mobilenet_tsne.png` | Two projections of the same DIP test crops. | Optional figure |
| `metrics/group_cv_folds.csv`, `metrics/group_cv_summary.csv`, `metrics/group_cv_membership.csv` | Fold-level metrics, mean/sample SD and per-image held-out assignment. | ML stability table plus audit appendix |
| `metrics/pj_matched_dataset_storage.csv`, `metrics/storage_metrics.csv` | PJ's exact matched-image bytes and AP's directory totals, respectively. The denominator and units differ. | Use matched figures in the main DIP table and label any directory totals explicitly |
| `storage_reduction/mobilenetv2_quantization.csv` | Original/INT8 file sizes and FP32-restored evaluation metrics. | DIP model-storage table |

## Provenance and remaining evidence

- **Raw vs DIP CNN** (roadmap pp. 16–19). The raw model achieved 95.36% clean accuracy and about 0.93 macro-F1. DIP achieved 89.40% and 0.8588 on the 151-image holdout. The roadmap reports about 35 and 13 seconds per epoch, respectively. These timing estimates are historical and lack a tracked epoch log or hardware benchmark. The raw checkpoint and raw per-image outputs were unavailable in the checked project files. Ask AP for `models/mobilenetv2_raw.pth` or the original `dl_predictions_raw.csv`, `dl_classification_report_raw.txt` and `dl_confusion_matrix_raw.png` produced by `python -m src.evaluate_dl_raw`.
- **Corruption stress test** (roadmap pp. 20–22). Use the paired numbers from the tracked CSV when citing that run (raw 95.36% → 34.44%, DIP 89.40% → 55.63%). The PDF gives 33.77% and 58.28% under noise. The script adds unseeded Gaussian noise (standard deviation 0.8) over the entire normalized tensor, including foreground. Treat this as synthetic corruption. Grad-CAM examples and this result motivate shortcut tests controlled by source or background.
- **Storage** (roadmap pp. 4–5, 21, 24). On the 1,004 matched paths, the dataset used 96.11 MB raw and 18.83 MB processed (80.40% saved). The directory-based script reports 282.56 and 26.87 under its MB label but divides bytes by 1,048,576 (MiB). Those totals include images beyond the paired census. The model file dropped from 9.144 to 2.654 decimal MB (70.98%).
- **Additional roadmap items** (pp. 3–5). ResNet and ONNX export, native INT8 inference, device benchmarks and independent field validation have no verified evidence in this snapshot. CNN/hybrid grouped CV would require fold-specific MobileNetV2 training.
- **Visual audit**. The existing three-model figure can be reproduced from the CSV matrix cells. Raw per-image records are needed to regenerate the raw matrix and compare exact errors crop by crop. The rounded roadmap metrics remain usable as clearly attributed reported values.

## Scope not implemented

| Item | Why it was in the plan | Status |
|---|---|---|
| ResNet50 | Second CNN for comparison | Dropped |
| DL inference time vs classical | Speed comparison for deployment | Dropped (classical timings exist: SVM ~35 ms, RF ~87 ms) |
| ONNX export | Cross-platform deployment | Dropped |
| Real INT8 quantization + edge benchmark | Latency/memory on device | Dropped (only weight-storage size measured) |

---
## Reference documents and historical provenance

All original PDFs are preserved without changing their contents:

- [Updated roadmap and CNN write-up](references/roadmap-MLEE-project.pdf), identified as the 24-page reference by the original inventory.
- [Earlier roadmap and dataset source links](references/roadmap-%20MLEE%20project.pdf).
- [Original pipeline plan](references/MLEE%20project%20pipeline.pdf).

The project originated at `ayush-1114/Cucumber-Disease-Pipeline` and moved to `shardulnarayan/mlee-project` on 17-18 November 2026. The original classical handoff records branch `classical-ml`, commit `902ffe6`, based on `4f499ad`, pushed 17 November. These are historical provenance, rather than current merge instructions. Processed CNN predictions were recovered from old-repo commit `41a46ea`; the reviewed updated roadmap was identified as blob `832b8997510dfb81f9c54e42f9c825915b107e56`.

The early local summary dated 16 November reported provisional SVM accuracy/macro-F1 89.4%/0.81 and RF 90.7%/0.79. These precede the canonical runs and must not replace the final tracked metrics. The original personal summaries and handoff documents are preserved in the local documentation backup.
