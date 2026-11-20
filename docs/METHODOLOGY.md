# Methodology

Integrated dataset, preprocessing, classical and deep-learning methods. See [Results](RESULTS.md) and [Reproducing](REPRODUCING.md). Code paths are relative to the repository root.

## Dataset construction

Notes for the report on how the 5-class dataset was assembled, cleaned and split.
Numbers come from `results/metrics/dataset_split_summary.csv`, written by
`src/data_prep/split_dataset.py`.

### Sources

No single public dataset covers all five classes, so the dataset combines several
sources (source links are preserved in [the earlier roadmap](references/roadmap-%20MLEE%20project.pdf)).

| Class | Source | How the images were obtained |
|---|---|---|
| Whitefly | Roboflow "Pests in Cucumber Plants" (YOLO format) | Bounding boxes cropped by `src/crop_pests.py` |
| Leaf_Miner | Roboflow "Pests in Cucumber Plants" (YOLO format) | Bounding boxes cropped by `src/crop_pests.py` |
| Red_Mite | Mendeley Indian Tea Leaves, "Red Spider" folder | Proxy class (tea leaves); capped at 1500 by `src/organize_datasets.py` |
| Yellow_Mosaic | Mendeley Luffa aegyptiaca, "Cucumber Mosaic Virus" folder | Proxy class; capped at 1500 |
| Healthy | Kaggle Cucumber Greenhouse dataset, "Healthy Leaves" folder | Copied as-is |

### Cleaning and downsampling

- **Exact duplicates dropped: 46 files** (same bytes, found by MD5 hash).
  41 were in Healthy (every Healthy image was present twice under two names) and
  5 in Yellow_Mosaic.
- **Downsampled to at most 300 images per class** (team decision, to reduce the
  imbalance and training time), chosen with the shared seed. This removed 1200
  Red_Mite, 1197 Yellow_Mosaic and 11 Whitefly images.
- Images that are not used are moved to `_unused/`, never deleted.
- After both steps: **1004 images**.

### Stratified 70/15/15 split

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Whitefly | 210 | 45 | 45 | 300 |
| Red_Mite | 210 | 45 | 45 | 300 |
| Yellow_Mosaic | 210 | 45 | 45 | 300 |
| Leaf_Miner | 43 | 10 | 10 | 63 |
| Healthy | 29 | 6 | 6 | 41 |
| **Total** | **702** | **151** | **151** | **1004** |

- **One shared list:** the split is defined by `data/split_manifest.csv`, which is
  committed. `python -m src.data_prep.split_dataset` applies that list to both
  `data/raw` and `data/processed`, so the classical ML, deep learning and evaluation
  all use the same train, val and test images.
- **Stratified:** the split is done class by class, so every class keeps the 70/15/15 ratio.
- **Leakage-safe:** the Whitefly crops come from only 65 photos and the 63 Leaf_Miner
  crops from only 18 photos. All crops of one photo are kept in the same split;
  otherwise near-identical crops would sit in both train and test and inflate the score.
- **Reproducible:** fixed seed (`SEED = 42` in `src/config.py`) and sorted file names,
  so rebuilding the list from the same `raw.zip` gives the identical list on any machine.

### Class imbalance

After downsampling, the largest classes (300) are still about 7 times the smallest (Healthy, 41).

- Classical models use `class_weight="balanced"`, and the small classes are augmented
  in the training set only (`src/classical_ml/augment.py`).
- Models are compared with **macro-F1**, not accuracy. A model that never predicts
  Healthy or Leaf_Miner would still reach about 89% accuracy.

### Dataset limitations

- **Healthy (41) and Leaf_Miner (63) are very small.** Their test sets have 6 and 10
  images, so one mistake moves per-class recall by 10-17 percentage points.
- **Each class comes from a different source**, with different crops, backgrounds and
  image sizes (Healthy: large photos on a white background; Whitefly / Leaf_Miner: small
  crops; Red_Mite: tea leaves, about a fifth of them black-and-white thresholded images). A model can partly separate classes by source style
  rather than by disease. Grad-CAM examples are available as an illustrative check. Independent source validation remains necessary.
- **Red_Mite and Yellow_Mosaic are proxy classes** from other plants (tea, luffa).
- Only exact duplicates were removed; near-duplicates (for example augmented copies
  inside a source dataset) were not checked.

## Development of the shared split

## Before

| Problem | Evidence |
|---|---|
| No split at all | `data/raw/val` and `data/raw/test` were empty; all 3458 images sat in `train/` |
| Hidden duplicates | 46 files were byte-identical copies (MD5 hash). **Every** Healthy image was there twice: 82 files, only 41 real images. 5 more in Yellow_Mosaic |
| Each person would split differently | The suggested fix was each person calling `train_test_split` on their own. Different seeds or file orders give different test sets, so SVM and MobileNetV2 scores could not be compared |
| Downsampling not repeatable | `src/downsample_data.py` shuffled files in the order the OS listed them, so "the 300 images kept" differed between laptops |
| Data leakage risk | Whitefly: 311 crops from only **65** photos. Leaf_Miner: 63 crops from only **18** photos. A per-image split puts crops of the same photo in both train and test |
| First attempt at grouping skewed the ratios | Grouping crops with `train_test_split` gave Whitefly **167 / 71 / 73 (54% / 23% / 23%)** instead of 70 / 15 / 15 |

## After

| Fix | Result |
|---|---|
| One committed list (`data/split_manifest.csv`) | Every teammate runs the same script and gets identical folders |
| Applies to both raw and processed | Verified: `data/raw` and `data/processed` both 702 / 151 / 151, same images |
| Duplicates removed | 3458 files to 3412 unique images (moved to `_unused/`, not deleted) |
| Repeatable downsampling to 300 per class | Same images kept on every machine (seeded, sorted file names) |
| Crops of one photo kept together | **0** photos split across train/val/test (checked on all groups) |
| Size-aware group assignment | Whitefly now **210 / 45 / 45 (70 / 15 / 15)** |
| Repeatable | Rebuilding the list twice gives a byte-identical file |

Final split:

| Class | Train | Val | Test |
|---|---|---|---|
| Whitefly | 210 | 45 | 45 |
| Red_Mite | 210 | 45 | 45 |
| Yellow_Mosaic | 210 | 45 | 45 |
| Leaf_Miner | 43 | 10 | 10 |
| Healthy | 29 | 6 | 6 |
| **Total** | **702** | **151** | **151** |

## DIP development

### Initial HSV masking attempt

DIP Pipeline Development: Iteration 1 (HSV Masking)

Methodology: Our initial pipeline utilized an HSV (Hue, Saturation, Value) color mask to isolate the green leaf, followed by GrabCut for background removal and Canny for edge detection.

Analysis of Failure: Upon visual inspection of the feature evolution plots, this approach proved functionally inadequate for multi-class disease detection:

Color Dependency: The HSV mask was tuned for healthy green leaves. When presented with a Whitefly (a white insect on a green leaf) or Red Mite (brown/yellow dry tea leaves), the mask inverted or failed entirely.

Cascading Failure: Because the HSV mask was broken, GrabCut received incorrect initializations, causing it to either segment the background instead of the leaf or crash entirely due to empty foreground/background samples.

Edge Noise: The Canny edge detector picked up high-frequency noise such as leaf veins, background grass, and image watermarks, creating false positives.

### Iteration 2: Otsu seeds

| Class | Failure |
|---|---|
| Yellow_Mosaic, Red_Mite | Output completely black: GrabCut removed the whole leaf |
| Whitefly | The white bug was treated as background; a black hole was punched where the bug was |
| Leaf_Miner | Only the dark hole was kept; the leaf was deleted |

Figures: `results/figures/Iteration_2_Otsu_Fail/*_evolution.png`

**Root causes**
- Otsu on greyscale separates *bright from dark*, not *leaf from background*. On a crop
  that is all leaf, it still splits something: the bug, or the hole.
- GrabCut was given only "probable" labels, so it was free to flip every pixel to
  background.
- The safety check only caught a completely black image, and a bare `except:` hid errors.

### Iteration 3: adaptive segmentation

| Change | Why |
|---|---|
| Seed GrabCut with an **HSV plant-colour mask** (hue brown-to-green, saturated) and fill holes | Colour separates leaf from a white/grey backdrop; filling keeps a white bug inside the leaf |
| **Skip GrabCut** when the mask covers > 85% (tight crop) or < 10% (unreliable), and always for Whitefly / Leaf_Miner crops | These crops have no background to remove |
| Give GrabCut **certain** foreground and background seeds, plus the padding as certain background | Stops it flipping everything to background |
| Fall back to the HSV mask if GrabCut keeps less than half of it | No near-empty outputs |
| **CLAHE after** segmentation | CLAHE changes the colours GrabCut relies on |
| **Otsu inside the leaf only**, on LAB a* | Now separates lesions (yellow/brown) from healthy green |
| Canny inside the eroded mask | The mask's own outline is not counted as an edge |

Test on 30 real images (6 per class):

| Class | Iteration 2 | Iteration 3 |
|---|---|---|
| Yellow_Mosaic | Black output | 5 segmented cleanly, 1 correctly kept whole |
| Red_Mite | Black output | 3 clean, 2 kept part of a dark background, 1 kept whole |
| Whitefly | Bug punched out | 4 correctly kept whole, 2 over-segmented (fixed by always skipping this class) |
| Leaf_Miner | Only the hole kept | All 6 correctly kept whole |
| Healthy | (worked) | All 6 segmented cleanly |
| **Black outputs** | **Several** | **0** |

Figures: `results/figures/*_evolution.png`

### Remaining segmentation limitations

- Some dark Red_Mite backgrounds are partly kept.
- The thresholds were hand-picked and checked on 30 images, not the full set.
- About a fifth of Red_Mite images are black-and-white, so colour segmentation cannot
  apply to them.

## Final DIP operations and diagnostic outputs

Script: `src/dip/pipeline.py`. Output: `data/processed/{train,val,test}/<class>/`, plus one
feature evolution plot per class (`results/figures/<Class>_evolution.png`:
Original → Mask → Final (CLAHE) → Otsu → Edges).

### The three iterations (before → after)

| Iteration | Method | What went wrong / right | Evidence |
|---|---|---|---|
| **1. HSV mask** | HSV green mask → GrabCut → Canny | Mask tuned for green leaves failed or inverted on Whitefly (white bug) and Red_Mite (brown tea leaves); bad mask gave GrabCut wrong seeds (wrong region or crash); Canny caught veins, grass, watermarks | `results/figures/Iteration_1_HSV_Fail/` + `it1-notes.txt` |
| **2. Otsu seed** | CLAHE → greyscale Otsu → GrabCut (probable labels only) | Yellow_Mosaic & Red_Mite fully black; Whitefly bug punched out as a hole; Leaf_Miner kept only the hole. Causes: Otsu splits bright/dark, not leaf/background; GrabCut free to flip every pixel; a bare `except:` hid errors | `results/figures/Iteration_2_Otsu_Fail/` |
| **3. Adaptive (final)** | See below | 30-image check (6/class): **black outputs several → 0** | `results/figures/*_evolution.png`, `docs/METHODOLOGY.md` |

### Iteration 3 steps and why

| Step | Detail | Why |
|---|---|---|
| Aspect-ratio pad + resize | 224 × 224, black padding | Uniform CNN input without distorting the leaf |
| HSV plant mask + hole filling | Hue 8–100 (brown→green), S > 35, V > 40, morphology | Separates leaf from white/grey backdrop; filling keeps a white bug inside the leaf |
| Adaptive GrabCut | Skipped if mask > 85% (tight crop) or < 10% (unreliable), and always for Whitefly / Leaf_Miner | Crops have no background to remove |
| Sure FG/BG seeds | Eroded mask = sure leaf; far from mask + padding = sure background | Stops GrabCut deleting everything |
| Fallback | Use HSV mask if GrabCut keeps < 50% of it | No near-empty outputs |
| CLAHE after segmentation | L channel of LAB, clip 2.0, 8×8 tiles | Enhances spots without disturbing colours GrabCut needs |
| Otsu inside leaf only | On LAB a* channel | Separates yellow/brown lesions from green; blank for Healthy |
| Canny inside eroded mask | Thresholds 50 / 150 | Mask outline not counted as an edge |

**Saved image** = CLAHE output × mask. Otsu and Canny outputs are for the plots and the report.

**Limitations:** thresholds hand-picked and checked on 30 images; some dark Red_Mite
backgrounds remain; black-and-white Red_Mite images defeat colour segmentation; Whitefly and
Leaf_Miner get no background removal.

---

## Applying and building the canonical split

**Run:** `python -m src.data_prep.split_dataset` (add `--dry-run` to preview only)

**The idea:** the split is one committed list, `data/split_manifest.csv`. It names
every image we use and says train, val or test. Everyone applies the same list, so
the SVM, the CNN and the evaluation all use identical images. Without this, each
person would make their own random split and the F1-scores could not be compared.

**Applying the list (the normal run)**

- For `data/raw` and `data/processed`, each image is matched to the list by class and
  file name and moved to `train/`, `val/` or `test/`.
- Images not in the list go to `_unused/`. Nothing is deleted.

**Building the list (`--rebuild`, used to establish the canonical manifest)**

1. **Collect** every raw image.
2. **Drop exact duplicates** using an MD5 hash (a fingerprint of the file's bytes).
   It found 46: every Healthy image was stored twice (41), plus 5 in Yellow_Mosaic.
3. **Downsample** each class to at most 300 images, with the shared seed.
4. **Group crops of the same photo.** Whitefly and Leaf_Miner images are crops cut
   out of Roboflow photos (63 Leaf_Miner crops from 18 photos). Crops of one photo
   are treated as one group.
5. **Split class by class** into 70% train, 15% val, 15% test, keeping each group
   whole. Groups are shuffled with the seed, then given one at a time (biggest first)
   to whichever split is furthest below its target.

Result: 702 train, 151 val, 151 test.

**Key terms**

- **Stratified:** each class is split separately, so every class has the same
  70/15/15 proportions. A plain random split could put all 41 Healthy images in train
  and none in test.
- **Data leakage:** information from the test set sneaking into training. Two crops
  of the same photo look almost identical; if one is in train and one in test, the
  model is tested on something it has effectively seen, and the score is inflated.
- **Seed:** the starting number for the random generator. Same seed, same shuffle,
  so the split is repeatable on any laptop.
- **Train / val / test:** train is for fitting the model, val is for choosing
  settings, test is touched once at the end for the final score.

**Design rationale**

- *Why not sklearn's `train_test_split(stratify=...)`?* It splits individual images,
  so it cannot keep crops of one photo together. With the groups it also drifted to
  54/23/23 for Whitefly, because groups have different sizes.
- *Why move duplicates instead of deleting them?* So the step can be undone and
  checked.
- *How do teammates get the same split?* They pull the committed manifest and run
  the script, which applies it to their raw and processed folders.

---

## Handcrafted feature extraction

**Run a check:** `python -m src.classical_ml.features path/to/image.jpg`

An SVM or Random Forest cannot read an image directly; it needs a fixed-length list
of numbers. Each image is resized to 224x224 and described by three feature types,
joined into one vector of **6208 values**.

| Feature | What it measures | Size | Why it helps here |
|---|---|---|---|
| **HOG** (Histogram of Oriented Gradients) | Shape: which edge directions occur where | 6084 | Leaf-miner tunnels, lesion outlines, leaf edges |
| **LBP** (Local Binary Patterns) | Fine texture | 28 | Mite speckling, mosaic mottling |
| **Colour histogram** (HSV) | How much of each colour | 96 | Yellowing vs healthy green |

**How each works**

- **HOG:** convert to grey, measure the edge direction at every pixel, then for each
  16x16-pixel cell count how many edges point in each of 9 directions. Cells are
  normalised in 2x2 blocks so lighting changes matter less.
  Size: 13 x 13 block positions x 4 cells x 9 directions = 6084.
- **LBP:** for each pixel, compare it with its neighbours on a circle: brighter = 1,
  darker = 0. That pattern of 0s and 1s is a code for the local texture. The feature
  is a histogram of the codes over the image. Two scales are used: 8 neighbours at
  radius 1 (10 bins) and 16 neighbours at radius 2 (18 bins) = 28.
  "Uniform" LBP only keeps the common simple patterns, which keeps it short.
- **Colour:** convert to HSV (Hue = which colour, Saturation = how strong,
  Value = how bright) and make a 32-bin histogram of each channel = 96. HSV is used
  instead of RGB because hue stays roughly the same when lighting changes.

**Design rationale**

- *Why hand-crafted features?* That is the point of Track 1: classical feature
  engineering (we choose the features) versus deep learning (the CNN learns its own).
- *Why 224x224?* It is `IMG_SIZE` in the config, the input size MobileNetV2 and
  ResNet50 expect, so both tracks see the same images.
- *HOG is 6084 of 6208 values. Does it drown out the others?* Partly, which is why
  the SVM pipeline standardises every feature and applies PCA first (see below).

---

## Training-only augmentation

**Save example picture:** `python -m src.classical_ml.augment`
(writes `results/figures/augmentation_examples_raw.png`)

**The problem:** even after downsampling, train has 210 images for each big class
but only 29 Healthy and 43 Leaf_Miner. The models learn to ignore the small classes.

**The fix:** every smaller class is topped up to the size of the largest class (210)
with randomly changed copies of its own training images:

- left-right and up-down flips
- a turn of 0 / 90 / 180 / 270 degrees, plus a small tilt of up to 25 degrees
- contrast x0.8 to x1.2, brightness -20 to +20

These are changes that do not alter the label: a flipped leaf-miner leaf is still a
leaf-miner leaf.

**The rules that keep it honest**

- **Train only.** Val and test are never augmented, so scores are measured on real
  images.
- **After the split.** Copies are made from train images only, so no copy of a test
  image can end up in train.
- **In memory.** Nothing is written into `data/`, so the split and AP's pipeline are
  unaffected.

**Design rationale**

- *Does augmentation add new information?* No. It shows the same leaves in new
  poses, which stops the model memorising orientation and lighting. It is not a
  substitute for collecting more real images.
- *Why also use `class_weight="balanced"`?* They do different jobs. Class weights
  make mistakes on rare classes cost more; augmentation gives the model more varied
  examples of them.

---

## SVM and Random Forest training

**Run:** `python -m src.classical_ml.train` (processed-input benchmark)
Options: `--dataset raw` (separate raw-input experiment), `--no-augment` (to compare), `--no-cache`.

**What it does, in order**

1. **Extracts features** for train, val and test, and caches them in
   `data/features/` so the next run is fast. If the split or the processed images
   change, run with `--no-cache`.
2. **Adds the augmented copies** to train.
3. **Tries a few settings for each model.** Each setting is fitted on train and
   scored on val; the best one is kept.
4. **Scores the chosen model once on test.**
5. **Saves** the model (`models/svm_model.pkl`, `models/rf_model.pkl`), the test predictions and a summary (scores, chosen settings,
   milliseconds per image) for PJ's evaluation.

**The two models**

- **SVM (Support Vector Machine)** - finds the boundary that separates the classes
  with the widest margin. The pipeline is:
  1. `StandardScaler`: puts every feature on the same scale (SVMs are sensitive to
     scale).
  2. `PCA` keeping 95% of the variance: compresses 6208 values to far fewer,
     removing redundancy. Fewer dimensions = faster and less overfitting.
  3. `SVC` with an RBF kernel: the kernel lets the boundary be curved.
  The setting tried is **C** (0.1, 1, 10, 100): small C = smoother boundary that
  tolerates some training mistakes; large C = fits the training data more tightly.
- **Random Forest** - 300 decision trees, each trained on a random sample of the
  data and features; the forest takes a vote. No scaling needed. The setting tried
  is **max_depth** (how many questions a tree may ask): unlimited, 10 or 20.

**Key terms**

- **Macro-F1:** F1 combines precision (of the images I called X, how many were X)
  and recall (of the real X images, how many I found). Macro-F1 is the plain average
  of F1 over the 5 classes, so Healthy counts as much as Red_Mite.
- **Why not accuracy?** A model that never predicts Healthy or Leaf_Miner still gets
  about 89% accuracy, because those classes are only 16 of 151 test images.
- **Hyperparameter:** a setting we choose (C, max_depth), as opposed to what the
  model learns from data. Choosing it on val, not test, keeps the test score honest.

**Design rationale**

- *Why must the final run use Dataset B?* The project compares classical ML against
  deep learning on the same DIP-processed images. Features from raw images would
  make that comparison invalid.
- *Why is inference timed?* PJ compares speed (ms per image) of classical ML versus
  deep learning for edge deployment. The timing covers the whole path: read the
  image, extract features, predict.
- *Why is the SVM "one model" when there are 5 classes?* sklearn's SVC trains one
  SVM per pair of classes (10 pairs) and takes a vote.

---

## Additional source observations

- Healthy (41 images) and Leaf_Miner (63) are tiny. Their test sets have 6 and 10
  images, so per-class scores for them are rough.
- Each class comes from a different source and looks different in style, so a model
  can partly tell classes apart by source rather than by disease. For example, roughly a fifth of the
  Red_Mite images (32 of a 150-image sample) are black-and-white thresholded images,
  which almost never occurs in the other classes. Grad-CAM examples are available as an illustrative check.
- Red_Mite and Yellow_Mosaic are proxy classes from other plants (tea, luffa).


## Model mechanisms and implementation details

**Why group and stratify?** Image-level random splitting can put near-identical crops from one photograph into training and test. A group is indivisible. Stratification attempts to maintain class proportions under that constraint. Exact byte hashing also removes identical files copied under different names. Counts and coverage matter separately: 63 Leaf Miner crops come from 18 source photos.

**HOG (6,084 values).** Convert to grayscale, estimate nearby intensity derivatives and vote for edge angles weighted by gradient strength. A 224 × 224 input has 14 × 14 cells of size 16 × 16. Adjacent 2 × 2 cells form a block, yielding 13 × 13 overlapping blocks with 4 × 9 bins each. L2-Hys normalizes each block, clips large bins and renormalizes. Edge directions can describe leaf tunnels or outlines. This representation has thousands of coordinates and can also capture incidental leaf texture.

**LBP (28 values).** Compare each neighbor with the center pixel, encoding brighter-or-equal neighbors as 1. A uniform LBP histogram summarizes simple patterns and puts less regular patterns into a common bin. The 8-neighbor radius-1 and 16-neighbor radius-2 histograms have 10 and 18 bins. This is a compact summary of local speckling or mottling.

**HSV (96 values).** Count pixels in 32 bins per channel for hue, saturation and value and normalize each histogram. It measures the distribution of colours and intensity without spatial location. Black mask pixels and capture conditions can influence these counts. HOG + LBP + HSV has 6,208 values.

**SVM.** A soft-margin support vector machine balances a wide separating margin with penalties for errors. The RBF kernel measures similarity as exp(-gamma * ||x - x'||^2), allowing curved boundaries. Larger C penalizes training violations more. Standardization gives coordinates comparable scale and PCA compresses correlated directions to retain 95% of training variance before fitting SVC. C is selected with validation macro-F1. PCA retains variance, so useful rare-class distinctions can still be lost.

**Random Forest.** Train 300 decision trees on bootstrap samples, using candidate feature subsets at each split, then average predicted class probabilities. Gini impurity measures class mixture. Tree splits can exploit nonlinear feature interactions at different scales. Balanced weights increase the influence of minority-class observations. Augmentation adds transformed Healthy and Leaf Miner training examples and preserves the untouched validation/test sets. It increases appearance variation within known examples. Independent source photos would provide additional diversity.

### Segmentation and MobileNetV2


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



## Evaluation interpretation

1. **Why report macro-F1 alongside accuracy?** Accuracy counts every correct crop equally, so large classes dominate. Precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is 2PR/(P+R). Macro-F1 averages the five class F1 values equally. Read it alongside support and recall.
2. **What does a confusion cell of 14 at row Whitefly and column Leaf Miner mean?** Fourteen true Whitefly crops were predicted Leaf Miner. This lowers Whitefly recall and Leaf Miner precision. Eight correct Leaf Miner crops out of ten give 80% recall.
3. **Does the largest test macro-F1 determine the winning ablation?** Rank the feature choices with validation data. Reserve the test result for describing the chosen configuration and explicitly label comparisons across all observed test outcomes.
4. **Can the fold SD be quoted as a confidence interval?** It describes variability across five overlapping training folds. The same dataset influenced the prior configuration choices. Nested evaluation or a fresh set would better estimate the whole selection procedure.
5. **Why skip grouped CV for the existing CNN features?** The saved checkpoint was built on a particular training split. Reusing it across new folds lets some held-out samples influence the feature extractor's state. Train a separate extractor/head per fold for a grouped CNN or hybrid evaluation.
6. **Can you claim the processed model works better in the field?** The processed model performed better under the tracked synthetic noise condition, and raw performed better on clean shared-source crops. Neither experiment samples independent farms, source pipelines and crop species.
7. **How is the DIP storage percentage calculated?** Use 100 × (1 − processed bytes/raw bytes) on the same set of paths. The 80.40% figure concerns images. The 70.98% figure concerns the weight-storage file.

## Source selection and initial assembly

The initial plan considered the Omer and Tencent datasets and the 2.2 GB WMC-Leafset; smaller sources were used instead. The original Red_Mite source had 10,109 images, initially capped at 1,500 before the canonical 300-per-class cap. Dataset assembly initially had 3,458 files, reduced to 3,412 byte-unique images before final downsampling. These are historical assembly counts, rather than the final 1,004-image experiment size.

The corrected segmentation logic was diagnosed and supplied by Shardul and integrated into the shared pipeline by Ayushman. Original source acquisition and CNN experiments were supplied by Ayushman; grouped splitting and classical modelling by Shardul; subsequent matched evaluation and diagnostics by Paarth. This shared provenance applies across the integrated workflow.
