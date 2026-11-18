# Progress 4 — Model Evaluation and Storage

**Owner** PJ · **Date** 17 Nov 2026

| Work | Result | Evidence |
| --- | --- | --- |
| Matched test-set comparison | Verified prediction filenames and labels on 151 DIP-processed images for SN's augmented SVM and Random Forest and AP's MobileNetV2. Test macro-F1 was 0.8017, 0.8611, and 0.8588, respectively. MobileNetV2 detected 8 of 10 Leaf Miner images. | [Confusion matrices](../../results/figures/model_comparison_confusion_matrices.png) |
| Random Forest feature ablation | Compared seven subsets of SN's HOG, LBP, and colour features. LBP + colour led on validation macro-F1 (0.9063). The seven saved result rows matched a rerun after migration. | [Ablation CSV](../../results/metrics/rf_feature_ablation.csv) |
| MobileNetV2 embeddings and SVM | Combined AP's embeddings with an SVM. The hybrid reached 95.36% test accuracy and 0.8774 test macro-F1. AP's original head had higher test Leaf Miner recall (8/10 versus 3/10). | [Comparison CSV](../../results/metrics/mobilenet_svm_comparison.csv) |
| Leaf Miner learning curve and feature visualization | Measured validation recall as Leaf Miner training source groups increased and compared HOG and MobileNetV2 features on the same 151 test crops. | [Learning curve](../../results/figures/leaf_miner_learning_curve.png) and [t-SNE figure](../../results/figures/hog_mobilenet_tsne.png) |
| Dataset and checkpoint storage | The 1,004 matched raw images went from 96.11 MB to 18.83 MB after DIP (80.40% reduction). AP's MobileNetV2 checkpoint went from 9.144 MB to 2.654 MB with INT8 weight storage (70.98% reduction). Quantized weights were restored to FP32 for evaluation, so this measures storage only. | [Storage analysis](../PJ_RESULTS_AND_DISCUSSION.md#storage-before-and-after-dip) and [checkpoint CSV](../../results/storage_reduction/mobilenetv2_quantization.csv) |

See [Results and Discussion](../PJ_RESULTS_AND_DISCUSSION.md) for methods, commands, and limitations. See [checkpoint 5](05_pj_report_evidence_and_viva.md) for the raw-versus-DIP results and recovered prediction evidence.

- Five-fold source-group cross-validation now covers the three classical configurations. RF with LBP plus colour led on mean macro-F1 (0.8529 ± 0.0660). See the [CV summary](../../results/metrics/group_cv_summary.csv) and [report](../PJ_RESULTS_AND_DISCUSSION.md#five-fold-stratified-group-cross-validation).
