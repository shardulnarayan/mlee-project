# Project guide and final report inventory

PJ compiled this guide from the repository, the [24-page roadmap](../roadmap-MLEE-project.pdf), and saved results. The project classifies five labels from a mixed-source dataset of leaf images and pest crops. The pipeline is source data → grouped split → raw and DIP variants → classical or deep classifier → evaluation. See the [viva guide](TEAM_VIVA_GUIDE.md) for implementation credit and concepts.

## Repository map

| Path | Purpose |
| --- | --- |
| `README.md`, `requirements.txt`, `.gitignore` | Entry point and links, Python dependencies, and exclusions for data, environments and model artifacts. |
| `roadmap-MLEE-project.pdf`, `roadmap- MLEE project.pdf`, `MLEE project pipeline.pdf` | Updated 24-page roadmap, earlier roadmap, and original pipeline plan. Use tracked code and saved evidence for implemented behavior. |
| `data/split_manifest.csv`, `data/dummy_data/` | Canonical split, label, filename and source-group assignments, plus a placeholder for smoke-test data. |
| `data/raw/`, `data/processed/`, `data/raw/_unused/` | Local source and processed split trees, plus excluded originals retained by the split tooling. Bulk images stay outside Git. |
| `models/`, `notebooks/` | Local model states and optional notebooks. The tracked directories are placeholders. |
| `src/config.py`, `src/__init__.py` | Shared paths, five labels, image size and seed, plus Python package marker. Subpackage markers and `.gitkeep` files carry no application logic. |
| `src/crop_pests.py`, `src/organize_datasets.py`, `src/downsample_data.py` | YOLO box crops, dataset assembly, and earlier downsampling helper. For the shared cleaned dataset start with the canonical manifest. |
| `src/data_prep/split_dataset.py` | Builds or applies grouped split assignments, reports counts and can preview moves with `--dry-run`. |
| `src/dip/pipeline.py` | Letterboxes, estimates foreground, enhances luminance, saves masked colour images and renders stage previews. |
| `src/classical_ml/README.md`, `features.py`, `augment.py`, `train.py` in the classical directory | Commands, HOG/LBP/HSV descriptors, training-only class balancing, and SVM/RF fitting and evaluation. |
| `src/train_dl.py`, `src/train_dl_raw.py` | MobileNetV2 training on processed and raw inputs, respectively. |
| `src/evaluate_dl.py`, `src/evaluate_dl_raw.py` | Evaluate the corresponding checkpoint and export per-image predictions, text report and heatmap. |
| `src/gradcam.py`, `src/gradcam_raw.py` | Generate class activation previews for the two input/checkpoint variants. |
| `src/background_swap.py`, `src/storage_metric.py` | Whole-input Gaussian noise stress test and whole-directory size census. |
| `src/classical_ml/feature_ablation.py`, `mobilenet_svm.py` | Compare seven RF feature subsets and compare handcrafted SVM, embedding SVM and AP's head. |
| `src/classical_ml/leaf_miner_learning_curve.py`, `plot_feature_tsne.py` | Measure validation performance versus Leaf Miner training-group count and plot test feature neighborhoods. |
| `src/classical_ml/measure_quantized_storage.py`, `group_cross_validation.py` | Measure weight-storage savings with FP32 re-evaluation and run five-fold classical group CV. |
| `reports/dataset_construction.md`, `sn_code_walkthrough.md`, `PUSHED_classical_ml.md` | SN's dataset provenance, script walkthrough and classical run handoff. |
| `reports/PJ_RESULTS_AND_DISCUSSION.md`, `reports/progress/` | PJ's detailed evaluation narrative and reproducible team milestones. |
| `reports/dip_report_assets/`, `reports/ml_report_assets/` | Staging folders for DIP and ML report assets. The tracked folders contain placeholders. |
| `results/metrics/`, `results/figures/`, `results/storage_reduction/` | Numeric evidence, generated visuals, and the model size comparison. |

## Data and split for the Methods section

| Label | Source | Train | Val | Test |
| --- | --- | ---: | ---: | ---: |
| Whitefly | Roboflow cucumber pest YOLO crops | 210 | 45 | 45 |
| Red_Mite | Mendeley Indian Tea Leaves, red spider mite proxy | 210 | 45 | 45 |
| Yellow_Mosaic | Mendeley Luffa aegyptiaca CMV proxy | 210 | 45 | 45 |
| Leaf_Miner | Roboflow cucumber pest YOLO crops | 43 | 10 | 10 |
| Healthy | Kaggle greenhouse cucumber leaves | 29 | 6 | 6 |
| **Total** | 46 exact duplicates removed before the final split | **702** | **151** | **151** |

See [dataset construction](dataset_construction.md) and roadmap pp. 1–2, 8–9. Across the three splits, each original pest photo contributes crops to a single split. Crop counts are larger than independent photo counts. Tea and luffa origins and source-specific backgrounds limit field generalization claims.

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
