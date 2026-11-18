# Cucumber Disease Classification

Five-class leaf and pest-crop classification using classical ML and MobileNetV2. The shared grouped split contains 1,004 images (702 train, 151 validation, 151 test). Red Mite and Yellow Mosaic use tea and luffa proxy datasets.

| Member | Contribution |
| --- | --- |
| AP | Adaptive DIP, raw and processed MobileNetV2, Grad-CAM and noise evaluation |
| SN | Dataset engineering, grouped split, handcrafted features and classical baselines |
| PJ | Matched evaluation, ablations and hybrid analysis, grouped CV, storage and report evidence |

## Start here

- [Project guide and report inventory](reports/PROJECT_GUIDE_AND_REPORT_INVENTORY.md)
- [Team contributions and viva guide](reports/TEAM_VIVA_GUIDE.md)
- [PJ results and discussion](reports/PJ_RESULTS_AND_DISCUSSION.md)
- [24-page roadmap](roadmap-MLEE-project.pdf)
- [Progress checkpoints](reports/progress/) and [latest evidence update](reports/progress/05_pj_report_evidence_and_viva.md)

## Run locally

From the repository root in PowerShell, set up Python and dependencies.

```powershell
python -m venv mlee-env
.\mlee-env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the team's cleaned archive contents in `data/raw/` with the expected split/class layout. The shared manifest defines the filenames and destination splits. Preview before applying the split, then generate the processed set.

```powershell
python -m src.data_prep.split_dataset --dataset raw --dry-run
python -m src.data_prep.split_dataset --dataset raw
python -m src.dip.pipeline
```

Use the [classical ML commands](src/classical_ml/README.md) for feature extraction and SVM/RF. To evaluate the CNNs, obtain AP's checkpoints and run `python -m src.evaluate_dl` for `models/mobilenetv2_cucumber.pth` or `python -m src.evaluate_dl_raw` for `models/mobilenetv2_raw.pth`.

## Saved evaluation

| Evidence | Result |
| --- | --- |
| Shared processed holdout | Augmented SVM 0.8017, augmented RF 0.8611, MobileNetV2 0.8588 macro-F1 |
| Five-fold grouped classical CV | LBP + colour RF 0.8529 mean macro-F1, sample SD 0.0660 |
| Matched dataset storage | 96.11 MB raw → 18.83 MB processed, 80.40% reduction |
| MobileNetV2 checkpoint | 9.144 MB FP32 → 2.654 MB INT8 weight storage, 70.98% reduction |

See [results](results/) for measurements and figures. Bulk images and model weights are local inputs.
