# Cucumber Leaf Pest & Disease Classification (DIP + ML)

A 5-class image classifier for cucumber leaves (**Whitefly, Red_Mite, Yellow_Mosaic, Leaf_Miner,
Healthy**). It is built as a **two-track study**: a Digital Image Processing (DIP) pipeline feeds
(1) classical ML (hand-crafted features + SVM / Random Forest) and (2) deep learning
(MobileNetV2 transfer learning). An **ablation study** trains the same CNN on raw and on
DIP-processed images to measure what preprocessing actually does.

> **Headline result.** On the clean test set the raw-image CNN scores higher (95.4% vs 89.4%).
> Under heavy input noise the raw model collapses (to 34.4%) while the DIP model holds better
> (55.6%), and Grad-CAM shows the raw model attends to backgrounds. DIP also cuts dataset
> storage by 80.4% and per-epoch training time by about 2.7x. The best classical model
> (Random Forest, LBP + colour) matches or beats the CNN on macro-F1.

---

## 1. Team and roles

| Code | Person | Role | GitHub |
|---|---|---|---|
| **AP** | Ayushman | DIP pipeline, deep learning, Grad-CAM, project setup | `percy-1411` |
| **SN** | Shardul | Data engineering (dataset, dedup, shared split), classical ML | `shardulnarayan` |
| **PJ** | Paarth | Evaluation, analysis, storage metrics, cross-validation, results write-up | Paarth Juneja |

Dependency chain: SN's dataset/split → AP's processed dataset (Dataset B) → SN's classical
models and AP's CNNs → PJ's evaluation → reports.

---

## 2. Status at a glance

| Roadmap item | Owner | Status | Where |
|---|---|---|---|
| 1. 5-class dataset, cleaning, 70/15/15 split | AP, SN | Done | §4 |
| 2. DIP pipeline + feature evolution plots | AP (SN fixed Iter. 2) | Done (3 iterations) | §5 |
| 3a. Classical ML (HOG/LBP/colour, SVM, RF) | SN | Done | §6 |
| 3b. MobileNetV2 on processed images | AP | Done | §7 |
| 3c. Ablation: MobileNetV2 on raw images | AP | Done | §7 |
| 4.1 Metrics + confusion matrices | PJ, AP | Done | §8 |
| 4.2 Error analysis | PJ, AP | Partial (in roadmap PDF, not in repo files) | §7, §11 |
| 4.3 Grad-CAM (raw and processed) | AP | Done | §7 |
| Extra: feature ablation, hybrid CNN+SVM, learning curve, t-SNE, group CV, noise test | PJ, AP | Done | §7, §8 |
| 5.4 Storage reduction | PJ, AP | Done (two numbers, see §9) | §9 |
| ResNet50, DL inference time, ONNX export, real INT8 / edge benchmark | — | **Dropped from scope** | §10 |
| Reports, presentation, poster | All | In progress | — |

---

## 3. Timeline (from Git history)

The project began in `ayush-1114/Cucumber-Disease-Pipeline` and moved to
`shardulnarayan/mlee-project` (this repo) on 17–18 Nov 2026.

| Date | Who | What happened | Effect |
|---|---|---|---|
| 13–14 Nov | AP | Repo, folder layout, `src/config.py`, `.gitignore`, `requirements.txt` | One shared set of paths, image size (224), seed (42), class names |
| ~14–15 Nov | AP | `crop_pests.py`, `organize_datasets.py`, `downsample_data.py` | Raw 5-class dataset assembled (Red_Mite capped from 10,109 to 1,500, then all classes to 300) |
| 16 Nov | AP, SN | DIP Iterations 1 → 2 → 3 (old-repo PR #1) | Final adaptive pipeline; 0 black outputs (§5) |
| 16 Nov | SN | Shared split, old-repo PR #2 | Duplicates removed, leakage-safe identical split for everyone (§4) |
| 17 Nov | AP | Pipeline handles splits, MobileNetV2 training (old-repo PR #3) | Dataset B and first CNN |
| 17 Nov | SN | Classical ML + 3 progress records (old-repo PRs #4–#7) | Baseline: RF macro-F1 0.86 |
| 17 Nov | AP | DL evaluation, raw-image ablation (old-repo `feat/dip-dl`) | Raw 95% vs processed 89% |
| 17 Nov | PJ | Evaluation & storage analysis (this repo, PR #1) | Matched comparison, feature ablation, hybrid, t-SNE, storage |
| 18 Nov | AP | Noise (OOD) test, storage script, raw/processed Grad-CAM, DL scripts | Robustness argument for DIP |
| 18 Nov | PJ | Five-fold group cross-validation (PR #2) | Classical results now have mean ± SD |
| 18 Nov | AP | Updated roadmap PDF | — |

---

## 4. Dataset (Phase 1) — AP sourced, SN cleaned and split

### Sources

No public dataset covers all five classes, so five sources were combined. The original plan
(Omer + Tencent datasets, and the 2.2 GB WMC-Leafset) was dropped for smaller sources.

| Class | Source | How obtained | Note |
|---|---|---|---|
| Whitefly | Roboflow "Pests in Cucumber Plants" (YOLO) | Boxes cropped by `crop_pests.py` | 311 crops from 65 photos |
| Leaf_Miner | Same Roboflow set | Boxes cropped | 63 crops from 18 photos |
| Red_Mite | Mendeley Indian Tea Leaves, "Red Spider" | Copied, capped | **Proxy** (tea leaves); ~1/5 are black-and-white |
| Yellow_Mosaic | Mendeley Luffa aegyptiaca, "Cucumber Mosaic Virus" | Copied, capped | **Proxy** (luffa) |
| Healthy | Kaggle Cucumber Greenhouse, "Healthy Leaves" | Copied | Large photos on white background |

### Before → after (SN, `src/data_prep/split_dataset.py`)

| Problem before | Fix | Effect |
|---|---|---|
| No split; all 3,458 images in `train/` | One committed list `data/split_manifest.csv` | Identical 702 / 151 / 151 split for all three tracks |
| 46 byte-identical duplicates (every Healthy image twice) | MD5 dedup | 3,458 → 3,412 unique; Healthy is 41, not 82 |
| Downsampling depended on OS file order | Seeded, sorted selection to ≤300 per class | Same images kept on every machine |
| Crops of one photo could land in train and test (leakage) | Group by source photo | 0 photos cross splits |
| First grouped split skewed ratios (Whitefly 54/23/23%) | Size-aware group assignment | Whitefly 70/15/15 |
| Excess images at risk of deletion | Moved to `_unused/`, never deleted | Fully recoverable |

### Final split (1,004 images)

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Whitefly | 210 | 45 | 45 | 300 |
| Red_Mite | 210 | 45 | 45 | 300 |
| Yellow_Mosaic | 210 | 45 | 45 | 300 |
| Leaf_Miner | 43 | 10 | 10 | 63 |
| Healthy | 29 | 6 | 6 | 41 |
| **Total** | **702** | **151** | **151** | **1,004** |

Why downsample to 300: faster iteration on CPU, less dominance by one source, and the
imbalance ratio fell from roughly 1,500 : 41 to 300 : 41. All models are compared on
**macro-F1**, because a model that never predicts Healthy or Leaf_Miner still reaches about 89%
accuracy.

---

## 5. DIP pipeline (Phase 2) — AP; Iteration 3 logic by SN

Script: `src/dip/pipeline.py`. Output: `data/processed/{train,val,test}/<class>/`, plus one
feature evolution plot per class (`results/figures/<Class>_evolution.png`:
Original → Mask → Final (CLAHE) → Otsu → Edges).

### The three iterations (before → after)

| Iteration | Method | What went wrong / right | Evidence |
|---|---|---|---|
| **1. HSV mask** | HSV green mask → GrabCut → Canny | Mask tuned for green leaves failed or inverted on Whitefly (white bug) and Red_Mite (brown tea leaves); bad mask gave GrabCut wrong seeds (wrong region or crash); Canny caught veins, grass, watermarks | `results/figures/Iteration_1_HSV_Fail/` + `it1-notes.txt` |
| **2. Otsu seed** | CLAHE → greyscale Otsu → GrabCut (probable labels only) | Yellow_Mosaic & Red_Mite fully black; Whitefly bug punched out as a hole; Leaf_Miner kept only the hole. Causes: Otsu splits bright/dark, not leaf/background; GrabCut free to flip every pixel; a bare `except:` hid errors | `results/figures/Iteration_2_Otsu_Fail/` |
| **3. Adaptive (final)** | See below | 30-image check (6/class): **black outputs several → 0** | `results/figures/*_evolution.png`, `reports/progress/02_dip_pipeline_fix.md` |

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

## 6. Classical ML (Phase 3, Track 1) — SN

Code: `src/classical_ml/` (`features.py`, `augment.py`, `train.py`). Details:
`src/classical_ml/README.md`, `reports/sn_code_walkthrough.md`.

- **Features (6,208 per image):** HOG shape (6,084) + LBP texture (28) + HSV colour histograms (96).
- **SVM:** StandardScaler → PCA (95% variance) → RBF SVM, `class_weight="balanced"`, C ∈ {0.1, 1, 10, 100}.
- **Random Forest:** 300 trees, `class_weight="balanced"`, max_depth ∈ {None, 10, 20}.
- Settings chosen on **val** by macro-F1; **test** scored once.
- **Targeted augmentation (train only):** Leaf_Miner +167, Healthy +181 (flips, rotations, brightness/contrast).

### Before → after augmentation (151 test images)

| Model | Accuracy | Macro-F1 | Leaf_Miner found | Healthy found | ms/image |
|---|---|---|---|---|---|
| SVM, no aug | 90.7% | 0.798 | 2/10 | 5/6 | 36 |
| SVM, aug | 88.7% | 0.802 | 4/10 | 6/6 | 35 |
| RF, no aug | 89.4% | 0.746 | 1/10 | 5/6 | 93 |
| **RF, aug** | **93.4%** | **0.861** | **6/10** | **6/6** | 87 |

Effect: RF Leaf_Miner F1 0.15 → 0.67. SVM accuracy fell while rare-class recall doubled, which
is why macro-F1 is used. Trade-off: RF Healthy precision fell 0.71 → 0.60. Timing was on a busy
laptop.

---

## 7. Deep learning, ablation and Grad-CAM (Phase 3 Track 2, Phase 4.3) — AP

### Training recipe (identical for raw and processed)

MobileNetV2 pre-trained on ImageNet, feature layers frozen, new 5-class head. Adam (lr 0.001),
cross-entropy, 10 epochs, batch 16, `WeightedRandomSampler` for imbalance, train-time flips,
±20° rotation and colour jitter; the best epoch by val accuracy is saved.
Scripts: `src/train_dl.py` → `models/mobilenetv2_cucumber.pth`,
`src/train_dl_raw.py` → `models/mobilenetv2_raw.pth`.
The processed model peaked at **94.04% val accuracy at epoch 5**.

### Ablation: raw vs DIP-processed (same 151 test images)

| Class F1 | DIP-processed | Raw | Better |
|---|---|---|---|
| Healthy | 1.00 | 1.00 | Tie |
| Leaf_Miner | 0.50 (P 0.36, R 0.80) | 0.72 (P 0.60, R 0.90) | Raw |
| Red_Mite | 1.00 | 1.00 | Tie |
| Whitefly | 0.81 (R 0.69) | 0.92 (R 0.87) | Raw |
| Yellow_Mosaic | 0.99 | 1.00 | Raw |
| **Accuracy** | **89.4%** | **95.4%** | Raw |
| **Macro-F1** | **0.86** | **0.93** | Raw |
| Training time / epoch | ~13 s | ~35 s | DIP (2.7x faster) |

Why raw wins on clean data (team analysis): (1) DIP resizes to 224 before training, which may
blur tiny Whitefly / Leaf_Miner detail; (2) each class comes from a different source with a
distinctive background, so the raw model can use background as a shortcut, and the test set
shares those backgrounds. Main error in the DIP model: 14 of 45 Whitefly predicted as Leaf_Miner
(both are un-segmented crops).

### Robustness test (`src/background_swap.py`)

| Condition | Raw model | DIP model |
|---|---|---|
| Clean test set | 95.36% | 89.40% |
| Gaussian noise (σ = 0.8 on normalised input) | 34.44% | 55.63% |
| Drop | 60.9 pts | 33.8 pts |

Effect: the DIP model loses about half as much. **Caveats:** despite the file name, this adds
noise to the whole image; it does not swap backgrounds. The noise is unseeded, so reruns
differ (an earlier run gave 33.77% vs 58.28%).

### Grad-CAM (`src/gradcam.py`, `src/gradcam_raw.py`)

One heatmap per class for each model in `results/figures/gradcam/` and `gradcam_raw/`.
Team reading: the raw model attends to padding and image edges (e.g. Leaf_Miner, Yellow_Mosaic,
Healthy); the DIP model attends to the insect, the tunnel and the mosaic patches. One image per
class, so this is illustrative rather than statistical.

---

## 8. Evaluation and extra analysis (Phase 4) — PJ

Full write-up: `reports/PJ_RESULTS_AND_DISCUSSION.md`.

| Analysis | Result | Effect / takeaway |
|---|---|---|
| Matched 3-model comparison (151 test) | SVM 0.802, **RF 0.861**, MobileNetV2 0.859 macro-F1 | Best classical ≈ CNN; CNN best on Leaf_Miner (8/10) |
| RF feature ablation (7 subsets) | **LBP + colour**: val 0.906, test 96.7% acc with 124 features; HOG alone weakest | Colour does most of the work; HOG adds little (and colour may reflect source) |
| Hybrid: MobileNetV2 embeddings + SVM | 95.36% acc, 0.877 macro-F1, but Leaf_Miner 3/10 vs 8/10 for the CNN head | Best overall accuracy; worse on the rare class |
| Leaf_Miner learning curve (3→13 source groups) | Val recall 0% → 20% | More Leaf_Miner sources help; data is the bottleneck |
| t-SNE (HOG vs MobileNetV2 features) | CNN features separate Whitefly better; Leaf_Miner still mixes in | Visual support for the confusion |
| 5-fold source-group CV (classical) | RF LBP+colour **0.853 ± 0.066**; RF full 0.806 ± 0.031; SVM 0.771 ± 0.025. Leaf_Miner recall 16.7%–84.6% across folds | Ranking is stable; rare-class scores are very noisy |
| INT8 weight storage (checkpoint) | 9.144 → 2.654 MB (−71%), test 92.7% after restoring to FP32 | Storage only; INT8 execution not measured |

---

## 9. Storage reduction (Phase 5.4) — use one number

| Measurement | Raw | Processed | Reduction | What it counts |
|---|---|---|---|---|
| PJ, `measure_quantized_storage.py` | 96.11 MB | 18.83 MB | **80.40%** | Only the 1,004 matched image pairs, decimal MB |
| AP, `storage_metric.py` | 282.56 MiB | 26.87 MiB | 90.49% | Every file under `data/raw` and `data/processed`, including `_unused/` and `train_backup/` |

**Use 80.40%** in the reports: it compares the same images. The 90.49% figure inflates the raw
side with thousands of images that were never used. Model size: 9.144 MB (decimal, PJ) and
8.74 MiB (binary, AP) describe essentially the same checkpoint in different units.

---

## 10. Planned but dropped

| Item | Why it was in the plan | Status |
|---|---|---|
| ResNet50 | Second CNN for comparison | Dropped |
| DL inference time vs classical | Speed comparison for deployment | Dropped (classical timings exist: SVM ~35 ms, RF ~87 ms) |
| ONNX export | Cross-platform deployment | Dropped |
| Real INT8 quantization + edge benchmark | Latency/memory on device | Dropped (only weight-storage size measured) |

---

## 11. Known limitations and open items

- **Proxy classes** (Red_Mite on tea, Yellow_Mosaic on luffa) and **one source per class**: models
  can learn source style rather than disease.
- **Tiny test classes:** Healthy 6, Leaf_Miner 10; one image moves recall by 10–17 points. Group CV
  shows the spread.
- **Leaf_Miner ↔ Whitefly confusion** is the main error for every model.
- **Not in the repo:** trained weights (`models/*.pth`, `*.pkl`), raw/processed images, the DL
  per-image prediction CSVs and classification reports. DL results cannot be re-run without the
  checkpoints; share them (e.g. Drive) and link them here.
- DIP thresholds checked on 30 images only. Noise test unseeded. Classical timings on a busy laptop.

---

## 12. File and folder guide

| Path | Owner | What it does | Effect / output |
|---|---|---|---|
| `src/config.py` | AP | Paths, `IMG_SIZE`, `SEED = 42`, 70/15/15, class list | Single source of truth |
| `src/crop_pests.py` | AP | Crops Whitefly / Leaf_Miner YOLO boxes from `external_data/` | Classification-ready pest crops |
| `src/organize_datasets.py` | AP | Copies Red_Mite, Yellow_Mosaic, Healthy, cap 1,500 | Raw 5-class folders |
| `src/downsample_data.py` | AP | Old cap to 300 (OS-order dependent) | **Superseded** by `split_dataset.py`; do not run |
| `src/data_prep/split_dataset.py` | SN | Applies (or `--rebuild`s) the manifest to raw and processed | Identical leakage-safe split |
| `data/split_manifest.csv` | SN | 1,004 rows: split, class, filename, source group | The split itself |
| `src/dip/pipeline.py` | AP / SN | Iteration 3 DIP pipeline | `data/processed/`, evolution plots |
| `src/classical_ml/features.py` | SN | HOG + LBP + colour (6,208) | Feature vectors |
| `src/classical_ml/augment.py` | SN | Train-only augmentation of small classes | Balanced training set |
| `src/classical_ml/train.py` | SN | SVM + RF, val selection, test scoring (`--no-augment`, `--no-cache`, `--dataset raw`) | `models/*.pkl`, predictions, summary JSON |
| `src/train_dl.py` / `train_dl_raw.py` | AP | MobileNetV2 on processed / raw | `mobilenetv2_cucumber.pth` / `mobilenetv2_raw.pth` |
| `src/evaluate_dl.py` / `evaluate_dl_raw.py` | AP | Test report, confusion matrix, per-image CSV | `dl_classification_report*.txt`, `dl_predictions*.csv` (not committed) |
| `src/gradcam.py` / `gradcam_raw.py` | AP | Grad-CAM, one image per class | `results/figures/gradcam*/` |
| `src/background_swap.py` | AP | Gaussian-noise robustness test | `ood_background_swap_results.csv` |
| `src/storage_metric.py` | AP | Whole-folder sizes, model size | `storage_metrics.csv` |
| `src/classical_ml/feature_ablation.py` | PJ | RF on 7 feature subsets | `rf_feature_ablation.csv` |
| `src/classical_ml/mobilenet_svm.py` | PJ | CNN embeddings + SVM | `mobilenet_svm_comparison.csv` |
| `src/classical_ml/leaf_miner_learning_curve.py` | PJ | Recall vs Leaf_Miner source groups | `leaf_miner_learning_curve.*` |
| `src/classical_ml/plot_feature_tsne.py` | PJ | t-SNE of HOG vs CNN features | `hog_mobilenet_tsne.png` |
| `src/classical_ml/measure_quantized_storage.py` | PJ | Matched storage + INT8 checkpoint size | `mobilenetv2_quantization.csv` |
| `src/classical_ml/group_cross_validation.py` | PJ | 5-fold source-group CV | `group_cv_*.csv` |
| `results/figures/` | All | Evolution plots (+ failed iterations), augmentation examples, confusion matrices, Grad-CAM, t-SNE, learning curve | Report figures |
| `results/metrics/` | All | Split summary, classical summaries and predictions, ablation, CV, noise test, storage | Report tables |
| `results/storage_reduction/` | PJ | Checkpoint storage CSV | DIP report metric |
| `reports/dataset_construction.md` | SN | Sources, cleaning, split, limitations | Report text |
| `reports/progress/01–04_*.md` | SN, PJ | Before/after record of each stage | Evidence for the reports |
| `reports/PUSHED_classical_ml.md`, `sn_code_walkthrough.md` | SN | What was pushed; plain-language code guide | Viva prep |
| `reports/PJ_RESULTS_AND_DISCUSSION.md` | PJ | Results and discussion draft | Report section |
| `roadmap- MLEE project.pdf`, `roadmap-MLEE-project.pdf` | AP | Original and updated roadmap (updated one adds DL, ablation, Grad-CAM notes) | Plan and report drafts |
| `MLEE project pipeline.pdf` | AP | Two-track design, role split, dependencies | Project plan |
| `data/`, `models/`, `notebooks/` | — | Images, weights (Git-ignored), unused notebooks folder | Local only |

---

## 13. How to reproduce

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
python src/train_dl.py                           # MobileNetV2, processed
python src/train_dl_raw.py                       # MobileNetV2, raw (ablation)
python src/evaluate_dl.py
python src/evaluate_dl_raw.py
python src/gradcam.py
python src/gradcam_raw.py
python src/background_swap.py
python src/storage_metric.py
python -m src.classical_ml.feature_ablation
python -m src.classical_ml.mobilenet_svm
python -m src.classical_ml.leaf_miner_learning_curve
python -u -m src.classical_ml.plot_feature_tsne
python -u -m src.classical_ml.measure_quantized_storage
python -u -m src.classical_ml.group_cross_validation
```

Note: `config.CLASSES` is not alphabetical, but `torchvision.ImageFolder` sorts classes
alphabetically. Compare predictions by class name, not index.
