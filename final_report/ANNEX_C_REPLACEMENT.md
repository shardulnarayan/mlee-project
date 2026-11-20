# Annex C: supplementary evidence and reproducibility

## Report introduction

> Supplementary reproducibility code is provided in
> `final_report/supplementary/`. The package reuses the team's CNN
> transforms and DIP functions and adds seeded orchestration, manifest-aligned
> evaluation, quantitative foreground attention and synthetic camera-fault
> checks. Outputs are stored in `results/supplementary/annex_c_v1/`, with
> execution status recorded in `table_status.json`.

The package imports the team's `src/` functions and provides orchestration and
quantitative analysis. Cite measured output files for each completed experiment.

| Annex item | Evidence and action |
|---|---|
| Table 6: three-seed CNNs | Seeded training is pending in the supplementary output directory. Report mean/sample SD and the measured environment after completion. |
| Table 7: Grad-CAM foreground mass | Attention evaluation is pending in the supplementary output directory. The default mask is a foreground proxy; report its definition and exclusions. |
| Table 8: class metrics | Cite `results/metrics/pj_holdout_per_class.csv`, preserving single-checkpoint model labels. |
| Table 9: ablation and CV | Cite `rf_feature_ablation.csv` and `group_cv_summary.csv` in `results/metrics/`. |
| Camera faults | Report measured outputs with checkpoint identity, fault severity and operation order. Brightness scores require the brightness evaluation stage. |
| INT8 storage | Completed: `results/storage_reduction/mobilenetv2_quantization.csv` establishes 70.98% smaller weight storage. Native INT8 runtime remains unmeasured. |

## Supplementary evaluation protocol

We use the committed 1,004-image source-group manifest and fixed 702/151/151
partition. For raw, team DIP and label-free DIP inputs, we train MobileNetV2
with seeds 0, 1 and 2. The runner retains the team's transforms, weighted
sampling, ImageNet initialization, frozen feature parameters and trainable
five-class head. Batch-normalization running statistics update in training
mode as in the original code. Adam uses learning rate 0.001, batch size 16
and ten epochs. Validation accuracy selects the checkpoint.

Label-free DIP enables segmentation for every image without consulting its
class to select the path. A fixed OpenCV seed and separate PNG cache make the
evaluation protocol explicit and reproducible.
Quantitative Grad-CAM targets the true class and uses documented foreground
proxies or externally validated masks. Mask failures and zero CAM mass are
excluded with explicit counts. Pixel faults follow resized input representation
and precede normalization. Normalized-tensor noise is a separate experiment.
This is not field-camera validation or a demonstration of image restoration.

## Before submitting the report

Run the suite, review `table_status.json`, and populate Tables 6/7 from the
completed measurement outputs. Include seed count,
sample SD, mask eligibility, fault settings and measured timing scope. Keep
native INT8 speed and independent field validation under Future Work.

This is editable report text. It does not modify or regenerate the
already-exported report PDF.
