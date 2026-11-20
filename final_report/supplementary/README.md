# Annex C supplementary experiments

This package provides seeded CNN training, quantitative foreground-attention
analysis and synthetic camera-fault evaluation. It imports the team's CNN
transforms and DIP functions from `src/`. Per-run outputs and execution status
are recorded alongside the summary tables.

## Commands (repository root, activated project environment)

```powershell
python -m final_report.supplementary.run check
python -m unittest final_report.supplementary.test_evidence
python -u -m final_report.supplementary.run all
```

`all` trains nine models: raw, team DIP and label-free DIP, seeds 0/1/2,
ten epochs, batch 16, Adam 0.001 and weighted replacement sampling. Then it
evaluates attention and camera faults and assembles tables. It can take
substantial time. ImageNet weights may download if absent from the local cache.
The default is CPU, four threads. Add `--device cuda` for a supported GPU.

Stages can run separately with the same `--output`, `--variants` and `--seeds`:

```powershell
python -u -m final_report.supplementary.run train
python -u -m final_report.supplementary.run attention
python -u -m final_report.supplementary.run corruptions
python -m final_report.supplementary.run tables
```

Default output: `results/supplementary/annex_c_v1/`. Checkpoints and label-free
images are ignored; CSV/JSON measurements remain eligible for review and commit.
Original models/results are not overwritten. No command invokes Git.
Add `--resume` to training or `all` after interruption: only hash-verified
checkpoints with the identical recipe/environment are reused. If a checkpoint
was interrupted before its JSON metadata was saved, use a new `--output`.

For a functional trial, train with `--variants dip --seeds 0 --epochs 1
--output results/supplementary/smoke_test`; use the same selection and output
for evaluation stages. A one-epoch trial is not the report experiment.

## Table 6: training and run metadata

Only the 6,405-parameter head learns gradients. As in the team code,
`model.train()` still updates batch-normalization running state in the frozen
backbone. The first epoch reaching the highest validation accuracy selects the
checkpoint. Test data never selects epochs. Timing includes training data
loading and optimizer updates, excludes validation and initial ImageNet loading,
and synchronizes CUDA. Sample SD uses `ddof=1`; one run has an empty SD.

Manifest loading enforces the canonical 702/151/151 source-group partition and
excludes `_unused/` and `train_backup/`. Predictions keep filenames, groups and
class names. Checkpoint metadata records its SHA-256, manifest hash, class order,
seed, recipe, source hashes and environment.

Label-free DIP enables `process_image(..., segment=True)` for every class,
resets OpenCV's segmentation RNG to zero per image and stores separate PNGs.
The PNG format and fixed RNG seed are part of the evaluation protocol and are
recorded with the run configuration.

## Table 7: quantitative Grad-CAM

The true-class target uses `MobileNetV2.features[-1]`. Attention share is
nonnegative CAM mass inside the mask divided by total CAM mass. Default masks
use the team's HSV heuristic in each variant's 224x224 model-input coordinates,
without a class-dependent segmentation decision. They are **foreground proxies,
not ground-truth leaf annotations**. Only 10%-85% coverage is accepted. Invalid
masks and zero-mass CAMs are excluded and counted, never assigned zero attention.

The summary pools valid image/seed records. Its spread is not an uncertainty
estimate over independent source photos. Eligibility can differ between variants.
Always report exclusion counts and mask area with attention shares.

For externally validated masks, add `--mask-dir PATH`. Supply binary 224x224
PNGs in `PATH/VARIANT/CLASS/FILENAME.png`, e.g.
`raw/Healthy/healthy_0009.jpg.png`. They must match the model-input geometry.
A missing external mask is an error. Label each attention table with its mask
definition, eligibility criteria and exclusion counts.

## Camera faults: evaluation settings

Pixel faults follow representation and resize to 224x224, before normalization:

| Fault | Settings |
|---|---|
| Pillow GaussianBlur radius | 1, 3, 5 |
| JPEG quality, subsampling 2 | 95, 50, 20, 5 |
| Brightness multiplication | 0.25, 0.5, 1.5, 2.0 |
| Rotation, bilinear, black fill | -20 and +20 degrees |

These test classifiers on degraded representations, not DIP's ability to repair
a corrupted raw capture. Brightness is only an exposure proxy. Separately,
unclipped Gaussian noise acts on normalized tensors at sigma 0.1/0.4/0.8, using
noise seeds 0-4 and matching standard-normal fields across models. Summaries
first average noise repeats per model, then report mean/sample SD across model
seeds; 15 evaluations are not 15 independently trained models.

Report camera-fault scores with the checkpoint, fault severity, encoder and
operation order recorded in `camera_protocol.json`. Brightness scores require
execution of the brightness evaluation stage.

## Table assembly and outputs

| Output | Purpose |
|---|---|
| `seeded_runs.csv`, `table6_seeded_summary.csv` | Individual runs and mean/sample SD |
| `*_epochs.csv`, `*_predictions.csv`, `*_per_class.csv` | Selection, timing and prediction evidence |
| `checkpoints/*.pth`, `checkpoints/*.json` | Local weights and provenance, ignored |
| `attention_per_image.csv`, `table7_attention_summary.csv` | CAM shares, mask areas, exclusions |
| `attention_protocol.json` | Target, layer, masks and pooling scope |
| `camera_faults.csv`, `camera_faults_summary.csv` | Corruption scores and model-seed summary |
| `camera_fault_predictions.csv`, `camera_protocol.json` | Image-level evidence and exact settings |
| `table8_existing_holdout_per_class.csv` | Existing team holdout class metrics |
| `table9_existing_*.csv` | Existing RF ablation and group-CV evidence |
| `table_status.json` | Explicit pending/completed assembly status |

`tables` requires no ML dependencies or training. It copies existing Tables 8/9
evidence, preserving model labels. Table 8's single-checkpoint CNN must not be
confused with the three-seed CNN mean. Tables 6/7 remain pending until the
required training and attention measurements are available.
