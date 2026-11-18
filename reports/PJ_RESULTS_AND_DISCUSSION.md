# Results and Discussion

## Classification on DIP-processed images

The augmented SVM and Random Forest trained by SN and the MobileNetV2 trained by AP were compared on the same 151 DIP-processed test images. PJ matched each model's saved predictions to the test manifest by filename and true label. Accuracy is the fraction of correctly classified images. Macro-F1 is the unweighted mean of the five class-level F1 scores.

| Model | Correct | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| SVM with augmentation | 134/151 | 88.74% | 0.8017 |
| Random Forest with augmentation | 141/151 | **93.38%** | **0.8611** |
| MobileNetV2 | 135/151 | 89.40% | 0.8588 |

![Side-by-side confusion matrices for SVM, Random Forest, and MobileNetV2](../results/figures/model_comparison_confusion_matrices.png)

*Figure 1. Confusion matrices for the 151-image test set. Rows represent true classes, columns represent predicted classes, and the three panels use a shared color scale for image counts.*

Random Forest correctly classified six more images than MobileNetV2, an accuracy difference of 3.97 percentage points calculated from the counts. Their macro-F1 scores differed by 0.0023. The models had different error patterns. MobileNetV2 correctly identified 8 of 10 Leaf Miner images (80% recall), compared with 6 of 10 for Random Forest and 4 of 10 for SVM. MobileNetV2 also predicted Leaf Miner for 14 of 45 Whitefly images. With only 10 Leaf Miner examples, a single image changes that class's recall by 10 percentage points.

## Random Forest feature ablation

PJ compared all seven nonempty subsets of SN's HOG (edge directions), LBP (local texture), and HSV colour histogram features. Each run used the same manifest-verified DIP splits, SN's 300-tree Random Forest with class weighting and unlimited depth, and seed 42. Each fit used the same 702 original training images and 348 augmented copies (167 Leaf Miner and 181 Healthy). Each validation and test split contained 151 original images. The full-feature control reproduced SN's validation macro-F1 and test scores to the reported precision.

| Features | Dimensions | Validation macro-F1 | Test accuracy | Test macro-F1 | Test Leaf Miner recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| HOG | 6,084 | 0.6806 | 85.43% | 0.7092 | 2/10 |
| LBP | 28 | 0.6986 | 86.09% | 0.6728 | 1/10 |
| Colour | 96 | 0.8844 | 96.03% | 0.9267 | 7/10 |
| HOG + LBP | 6,112 | 0.7068 | 86.09% | 0.7163 | 2/10 |
| HOG + Colour | 6,180 | 0.8537 | 92.05% | 0.8282 | 4/10 |
| **LBP + Colour** | **124** | **0.9063** | **96.69%** | 0.9094 | 5/10 |
| HOG + LBP + Colour | 6,208 | 0.8578 | 93.38% | 0.8611 | 6/10 |

LBP + Colour led validation macro-F1 and improved test accuracy by 5 correct images over the full-feature control. Colour alone had the highest observed test macro-F1 and detected 7 of 10 Leaf Miner images. That test result is descriptive. The validation ranking determines the preferred subset. Colour gave the strongest single-feature result on validation, and adding HOG lowered validation macro-F1 for both subsets that contained colour. The Random Forest's feature sampling uses the square root of the number of input columns, so changing the subset also changes how many candidate columns each split considers.

These results come from one seed and one split. The dataset includes proxy classes from different plants or image sources, so the colour signal could partly reflect source or capture conditions. Generalization to cucumber field photos remains unmeasured.

Run from the repository root with `python -m src.classical_ml.feature_ablation`. See the [full metrics CSV](../results/metrics/rf_feature_ablation.csv) for all per-class recalls.

## MobileNetV2 embeddings and SVM

PJ ran the comparison on the manifest-verified DIP split. The two SVMs used the same 702 original training images without augmentation and selected C by macro-F1 on the 151 validation images. The handcrafted model used SN's HOG, LBP, and colour features. The hybrid used the 1,280 pooled features from AP's MobileNetV2 checkpoint. AP's trained head received the same embeddings and retained its original training recipe.

| Model | Validation macro-F1 | Test correct | Test accuracy | Test macro-F1 | Test Leaf Miner recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Handcrafted features + SVM | 0.8060 | 137/151 | 90.73% | 0.7984 | 2/10 |
| MobileNetV2 embeddings + SVM | 0.8302 | 144/151 | 95.36% | 0.8774 | 3/10 |
| MobileNetV2 embeddings + AP head | 0.9116 | 135/151 | 89.40% | 0.8588 | 8/10 |

The embedding SVM gained seven correct test predictions over the handcrafted SVM and improved validation and test macro-F1. Against AP's original head, the hybrid gained nine correct test predictions and classified all 45 Whitefly images correctly. All seven hybrid errors were Leaf Miner images. AP's head achieved the highest validation macro-F1 and detected five more Leaf Miner cases on test. The head and SVM differed in training recipe as well as classifier. These results describe the shared split.

The reconstructed checkpoint's head reproduced AP's reported aggregate test scores. The comparison CSV contains aggregate metrics, so per-image agreement was not checked. Run with `python -m src.classical_ml.mobilenet_svm`. See the [comparison CSV](../results/metrics/mobilenet_svm_comparison.csv) for all per-class recalls and selected C values.

## Leaf Miner learning curve

PJ fitted the embedding SVM with C = 1 on nested subsets of Leaf Miner source groups, keeping the other four training classes and AP's MobileNetV2 checkpoint fixed. Crops from each source stayed together. Each partial group count used five seeded draws. The full set was fitted once. Evaluation used the same validation split for all runs.

| Leaf Miner training source groups | Training crops (range) | Mean validation Leaf Miner recall | Mean validation macro-F1 |
| ---: | ---: | ---: | ---: |
| 3 | 7–16 | 0% | 0.7800 |
| 7 | 19–31 | 0% | 0.7791 |
| 10 | 32–39 | 6% | 0.7960 |
| 13 | 43 | 20% | 0.8302 |

![Validation recall and macro-F1 by Leaf Miner training group count](../results/figures/leaf_miner_learning_curve.png)

At 10 groups, three of five runs detected one Leaf Miner crop and the other two detected zero. With all 13 training groups, the SVM detected two out of ten validation crops. This suggests that broader Leaf Miner training coverage helped the SVM on this split, though recall remained low. The validation crops come from two source groups, and adding groups also changes the crop count and class balance. The experiment varied the training data available to the SVM with AP's existing feature extractor fixed.

See the [per-run CSV](../results/metrics/leaf_miner_learning_curve.csv).

## t-SNE visualization of test features

![Side-by-side t-SNE plots of HOG and MobileNetV2 features for 151 DIP-processed test crops](../results/figures/hog_mobilenet_tsne.png)

Both panels show the same 151 manifest-matched test crops, coloured by their true class. PJ reused SN's HOG descriptor and AP's 1,280-dimensional MobileNetV2 embeddings. Each feature matrix was reduced separately to 30 PCA components, then projected to 2D using t-SNE with perplexity 20 and seed 42. Orange diamonds mark Leaf Miner and blue triangles mark Whitefly.

The MobileNetV2 projection shows a more distinct Whitefly region, though some Leaf Miner crops still appear among Whitefly points. The independent 2D maps illustrate local neighborhoods. Use the measured scores and recalls to compare classifiers. The left panel uses HOG alone, while the handcrafted SVM comparison used HOG, LBP, and colour together.

Reproduce the figure with `python -u -m src.classical_ml.plot_feature_tsne`.

## Storage before and after DIP

PJ measured the saved file lengths of each raw and processed image pair, using the split, class, and filename in the [shared manifest](../data/split_manifest.csv). All 1,004 pairs were present and nonempty. The sums cover only manifest-listed files and exclude extra directories such as `_unused` and `train_backup`. The reduction is calculated as 100 × (1 − processed bytes / raw bytes).

| Split | Matched pairs | Raw bytes | DIP-processed bytes | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Train | 702 | 68,913,353 | 13,220,174 | 80.82% |
| Validation | 151 | 13,462,596 | 2,914,825 | 78.35% |
| Test | 151 | 13,730,467 | 2,699,992 | 80.34% |
| **Total** | **1,004** | **96,106,416** | **18,834,991** | **80.40%** |

The processed dataset used 18.83 MB compared with 96.11 MB for the same raw files, a saving of 77.27 MB (1 MB = 1,000,000 bytes). These figures reflect the full preprocessing pipeline and its saved outputs, including resizing and encoding. The contribution of individual preprocessing stages to the storage change was unmeasured.

The three-model comparison uses processed inputs. AP's raw-versus-DIP MobileNetV2 comparison is available in [the raw/DIP table](../results/metrics/pj_cnn_raw_vs_dip.csv) and interpreted below.

## MobileNetV2 model storage before and after quantization

PJ measured AP's original checkpoint against an INT8 weight-storage version of the same model. The script quantizes each convolution and linear weight tensor with a separate symmetric scale for each output channel. Other state, including biases and batch-normalization parameters, keeps its original dtype. Both measurements use decimal MB (1,000,000 bytes). Neither measured file is an outer ZIP download.

| Checkpoint | Bytes | Size (MB) | Reduction | Test accuracy | Test macro-F1 | Leaf Miner recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original FP32 | 9,144,076 | 9.144 | — | 89.40% (135/151) | 0.8588 | 8/10 |
| INT8 weight storage | 2,654,043 | 2.654 | 70.98% | 92.72% (140/151) | 0.8920 | 8/10 |

The stored checkpoint saved 6.490 MB. Both evaluations used the same 151 DIP-processed test crops. The INT8 weights were restored to FP32 before inference because the installed PyTorch build lacked a supported INT8 execution backend. These measurements cover checkpoint storage and test predictions after rounding the weights. INT8 execution speed and peak memory were unmeasured. The higher accuracy is an observation on one test split.

Reproduce with `python -u -m src.classical_ml.measure_quantized_storage`. See the [measurement CSV](../results/storage_reduction/mobilenetv2_quantization.csv).

## Five-fold stratified group cross-validation

PJ cross-validated three previously selected classical configurations on the 1,004 DIP-processed images. The splitter used the manifest's source groups with seed 42. Every image was held out exactly once, no source group crossed a fold boundary, and every held-out fold contained all five classes. Each fold contained 200–201 held-out crops, including 8–9 Healthy crops and 12–13 Leaf Miner crops. Balancing augmentation and SVM scaling and PCA were fitted within each training fold.

| Configuration | Accuracy (mean ± SD) | Macro-F1 (mean ± SD) | Healthy recall (mean ± SD) | Leaf Miner recall (mean ± SD) |
| --- | ---: | ---: | ---: | ---: |
| SVM, HOG + LBP + colour | 89.44% ± 0.90 pp | 0.7709 ± 0.0247 | 80.83% ± 18.07 pp | 24.23% ± 19.83 pp |
| Random Forest, HOG + LBP + colour | 90.83% ± 2.77 pp | 0.8061 ± 0.0305 | 88.33% ± 13.63 pp | 31.79% ± 13.06 pp |
| Random Forest, LBP + colour | 93.02% ± 3.16 pp | 0.8529 ± 0.0660 | 90.56% ± 9.49 pp | 45.77% ± 26.42 pp |

The LBP + colour Random Forest had the highest mean macro-F1 among the three configurations. Its Leaf Miner recall ranged from 16.67% to 84.62%, showing how strongly recall varied with the held-out groups despite consistently high overall accuracy. The sample SD describes variation across these five folds. The configurations were chosen in earlier analyses on this same dataset, so this is a stability check of fixed configurations. An independent assessment of model selection would require nested CV or fresh data. AP's MobileNetV2 checkpoint has already seen some would-be held-out images, so CNN and hybrid cross-validation require fold-specific training.

Run `python -u -m src.classical_ml.group_cross_validation`. See the [fold metrics](../results/metrics/group_cv_folds.csv), [mean and SD for all metrics](../results/metrics/group_cv_summary.csv), and [fold membership](../results/metrics/group_cv_membership.csv).
## Raw versus DIP MobileNetV2

AP's raw model reached 95.36% clean test accuracy, compared with 89.40% for the DIP model. Macro-F1 was about 0.93 raw (rounded roadmap value) and 0.8588 DIP (recomputed from AP's saved predictions). The roadmap reports approximately 35 s per epoch for raw and 13 s for DIP. These are historical training measurements. The tracked synthetic Gaussian corruption run reached 34.44% raw and 55.63% DIP. The noise covered the whole normalized input. See the [evidence reconciliation](PROJECT_GUIDE_AND_REPORT_INVENTORY.md#provenance-and-remaining-evidence) for the differing roadmap corruption numbers and the raw checkpoint handoff.
