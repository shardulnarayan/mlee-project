# Team contributions and viva guide

Prepared by PJ. Each section credits the person responsible for the implemented work. See the [repository and evidence guide](PROJECT_GUIDE_AND_REPORT_INVENTORY.md) for result paths and the [roadmap](../roadmap-MLEE-project.pdf) for the original phases and AP's CNN write-up.

## SN · Dataset engineering and classical ML

**What SN did.** SN documented how the five source labels were acquired, removed exact duplicates, downsampled the majority classes, and produced the shared split and its manifest. PNG/JPG crops of the same source photo are grouped together. The final splits contain 702 training, 151 validation and 151 test crops. Whitefly and Leaf Miner come from YOLO bounding boxes. Converting a normalized box center and size to pixel bounds yields an insect or lesion close-up. The tea and luffa classes are proxies because the team needed compatible visual labels across multiple datasets.

**Why group and stratify?** Image-level random splitting can put near-identical crops from one photograph into training and test. A group is indivisible. Stratification attempts to maintain class proportions under that constraint. Exact byte hashing also removes identical files copied under different names. Counts and coverage matter separately: 63 Leaf Miner crops come from 18 source photos.

**HOG (6,084 values).** Convert to grayscale, estimate nearby intensity derivatives and vote for edge angles weighted by gradient strength. A 224 × 224 input has 14 × 14 cells of size 16 × 16. Adjacent 2 × 2 cells form a block, yielding 13 × 13 overlapping blocks with 4 × 9 bins each. L2-Hys normalizes each block, clips large bins and renormalizes. Edge directions can describe leaf tunnels or outlines. This representation has thousands of coordinates and can also capture incidental leaf texture.

**LBP (28 values).** Compare each neighbor with the center pixel, encoding brighter-or-equal neighbors as 1. A uniform LBP histogram summarizes simple patterns and puts less regular patterns into a common bin. The 8-neighbor radius-1 and 16-neighbor radius-2 histograms have 10 and 18 bins. This is a compact summary of local speckling or mottling.

**HSV (96 values).** Count pixels in 32 bins per channel for hue, saturation and value and normalize each histogram. It measures the distribution of colours and intensity without spatial location. Black mask pixels and capture conditions can influence these counts. HOG + LBP + HSV has 6,208 values.

**SVM.** A soft-margin support vector machine balances a wide separating margin with penalties for errors. The RBF kernel measures similarity as exp(−γ‖x − x′‖²), allowing curved boundaries. Larger C penalizes training violations more. Standardization gives coordinates comparable scale and PCA compresses correlated directions to retain 95% of training variance before fitting SVC. C is selected with validation macro-F1. PCA retains variance, so useful rare-class distinctions can still be lost.

**Random Forest.** Train 300 decision trees on bootstrap samples, using candidate feature subsets at each split, then average predicted class probabilities. Gini impurity measures class mixture. Tree splits can exploit nonlinear feature interactions at different scales. Balanced weights increase the influence of minority-class observations. Augmentation adds transformed Healthy and Leaf Miner training examples and preserves the untouched validation/test sets. It increases appearance variation within known examples. Independent source photos would provide additional diversity.

## AP · DIP pipeline and MobileNetV2

**What AP did.** AP built the DIP iterations, trained/evaluated raw and processed MobileNetV2, produced Grad-CAM examples and ran Gaussian-noise corruption. The [roadmap](../roadmap-MLEE-project.pdf) pp. 10–22 records the iterations, CNN procedure and observed results.

**DIP stages and design choices.**

| Stage | Mechanism and role |
| --- | --- |
| Resize and pad | Fit the longer side into 224 pixels, keep aspect ratio and record valid pixels. This creates consistent dimensions with less geometric distortion. |
| HSV prior and morphology | Threshold blur-smoothed HSV for plant-coloured regions. Closing fills small gaps, opening removes isolated specks, and contour filling fills holes in large regions. A green background or diseased discolouration can mislead this prior. |
| GrabCut | Fit foreground/background colour models and minimize a graph-cut objective balancing those models and nearby pixel consistency. The HSV prior seeds likely regions. Erosion supplies sure foreground and dilation supplies sure background beyond the prior. Coverage/support checks allow fallback to the valid area or prior. Whitefly and Leaf Miner take the valid-area path because they are known tight crops in this labelled dataset. A new-image pipeline would need a label-independent crop decision. |
| CLAHE | Equalize the LAB lightness channel locally and clip each tile histogram to limit noise amplification. This increases local contrast while preserving chroma channels. The saved classifier input is the enhanced colour image masked by foreground. |
| Otsu and Canny | Otsu selects a global threshold from foreground LAB a-channel values by maximizing between-class variance. Canny smooths, finds gradients, thins edges and uses two thresholds with connectivity. Its output is restricted to eroded foreground. Both are diagnostic stage images. |

**CNN architecture.** MobileNetV2 uses depthwise convolution to filter each channel spatially, then pointwise 1 × 1 convolutions to mix channels. An inverted residual block expands to a wider hidden representation, filters it depthwise, and projects back through a linear bottleneck. Where shape and stride allow, an identity skip preserves information. Global average pooling converts final feature maps to 1,280 features. A linear head maps those features to five logits. Cross-entropy penalizes low probability assigned to the true class.

**Training recipe.** The ImageNet-initialized feature parameters are frozen and Adam updates the replacement classifier. If batch normalization stays in train mode, running statistics can still update. Inverse-frequency sampling increases minority-class exposure. Training flips, rotates and jitters images. Validation/test resize and use ImageNet normalization without augmentation. The scripts run ten epochs, choosing the checkpoint by validation accuracy. Track this criterion separately from SN's validation macro-F1.

**Interpretation.** AP's raw model led on clean accuracy (95.36% versus 89.40% for DIP). The tracked Gaussian-noise test showed a larger decline for raw. Those variants differ in preprocessing and their separately trained checkpoints, so the comparison measures the end-to-end variants. Grad-CAM weights spatial convolution maps by gradients of a target score, highlights regions supporting the score, and upsamples the map. A visual example can suggest a shortcut or lesion focus. Testing independent field photos or manipulating backgrounds while preserving lesions would address those hypotheses directly.

## PJ · Matched evaluation, diagnostics and reporting

| Contribution | Method, rationale and result |
| --- | --- |
| Shared test comparison | Joined SN's and AP's predictions on filename and true class and verified 151 holdout crops. Augmented SVM, RF and CNN scored 0.8017, 0.8611 and 0.8588 macro-F1. AP's CNN caught 8 of 10 Leaf Miner crops, with 14 Whitefly crops misclassified as Leaf Miner. |
| RF feature ablation | Retrained the same forest on each of seven nonempty HOG/LBP/HSV subsets with the same augmented training data. LBP + colour led by validation macro-F1 (0.9063). Colour alone led observed test macro-F1 (0.9267). Choosing by validation avoids tuning on the test scores. Forest feature subsampling also changes when the feature-vector width changes. |
| Hybrid embedding SVM | Extracted AP's frozen checkpoint's 1,280-value vectors and compared an SVM on those vectors against SN's handcrafted features under matched training data. Test accuracy rose from 90.73% to 95.36% and macro-F1 from 0.7984 to 0.8774. The hybrid caught 3/10 Leaf Miner crops compared with AP's head at 8/10. The original head used a different training recipe. |
| Leaf Miner learning curve | Varied only the available Leaf Miner training groups (3, 7, 10, 13) for a fixed embedding SVM. The first two yielded 0% validation recall, 10 groups averaged 6%, and all 13 yielded 20%. Groups matter more than treating repeated crops as independent; the number of groups and crops both changed. |
| t-SNE | Applied PCA to 30 dimensions then t-SNE to each feature representation for the same 151 test crops. t-SNE emphasizes local neighbor similarities in a 2D layout. The plots help inspect overlap but distances between clusters or across the two layouts lack a common scale. |
| Storage | Counted original and processed bytes over the same 1,004 manifest paths. The saved DIP images used 80.40% fewer bytes. Quantized convolution/linear weights with a symmetric scale per output channel. Stored integer weights reduced checkpoint size by 70.98%. The evaluation decoded weights back to FP32, so the experiment measured storage, with no native INT8 latency evidence. |
| Grouped cross-validation | Stratified source groups across five folds. Training-only augmentation, scaling and PCA were re-fit in each fold. RF LBP + colour reached mean macro-F1 0.8529 with sample SD 0.0660. Leaf Miner recall averaged 45.77% with SD 26.42 percentage points. This quantifies sensitivity to split composition for previously chosen settings. |
| Documentation and result recovery | Built the inventory and exported confusion counts, per-class scores and holdout comparisons from five manifest-checked processed prediction files. Transcribed AP's rounded raw metrics with source attribution and recorded evidence needed for an exact raw error audit. |

### Viva questions

1. **Why report macro-F1 alongside accuracy?** Accuracy counts every correct crop equally, so large classes dominate. Precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is 2PR/(P+R). Macro-F1 averages the five class F1 values equally. Read it alongside support and recall.
2. **What does a confusion cell of 14 at row Whitefly and column Leaf Miner mean?** Fourteen true Whitefly crops were predicted Leaf Miner. This lowers Whitefly recall and Leaf Miner precision. Eight correct Leaf Miner crops out of ten give 80% recall.
3. **Does the largest test macro-F1 determine the winning ablation?** Rank the feature choices with validation data. Reserve the test result for describing the chosen configuration and explicitly label comparisons across all observed test outcomes.
4. **Can the fold SD be quoted as a confidence interval?** It describes variability across five overlapping training folds. The same dataset influenced the prior configuration choices. Nested evaluation or a fresh set would better estimate the whole selection procedure.
5. **Why skip grouped CV for the existing CNN features?** The saved checkpoint was built on a particular training split. Reusing it across new folds lets some held-out samples influence the feature extractor's state. Train a separate extractor/head per fold for a grouped CNN or hybrid evaluation.
6. **Can you claim the processed model works better in the field?** The processed model performed better under the tracked synthetic noise condition, and raw performed better on clean shared-source crops. Neither experiment samples independent farms, source pipelines and crop species.
7. **How is the DIP storage percentage calculated?** Use 100 × (1 − processed bytes/raw bytes) on the same set of paths. The 80.40% figure concerns images. The 70.98% figure concerns the weight-storage file.
