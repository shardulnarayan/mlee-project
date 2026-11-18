# Progress 5 · PJ report evidence and viva

**Owner** PJ · **Date** 18 November 2026

| Work | Verified evidence |
| --- | --- |
| Project and report inventory | [Project guide](../PROJECT_GUIDE_AND_REPORT_INVENTORY.md) indexes every tracked result family and maps it to the DIP/ML report. The GitHub roadmap blob `832b8997510dfb81f9c54e42f9c825915b107e56` is the reviewed 24-page version. |
| Team technical walkthrough | [Viva guide](../TEAM_VIVA_GUIDE.md) credits SN's split and classical work, AP's DIP and CNN work, and PJ's analyses and reporting. |
| Recovered prediction evidence | Restored AP's processed MobileNetV2 predictions from old team repo commit `41a46ea`. Verified exactly 151 unique manifest-matching keys for each of five processed test runs. [Aggregate](../../results/metrics/pj_holdout_comparison.csv), [per class](../../results/metrics/pj_holdout_per_class.csv), [matrix counts](../../results/metrics/pj_holdout_confusion_counts.csv). |
| Report tables | [Matched storage](../../results/metrics/pj_matched_dataset_storage.csv), [raw CNN class results](../../results/metrics/pj_raw_cnn_roadmap_metrics.csv), [raw versus DIP](../../results/metrics/pj_cnn_raw_vs_dip.csv). Raw CNN scores have explicit roadmap precision and provenance. |
| Evidence reconciliation | Tracked corruption CSV and PDF represent differing noise readings. Label the tracked CSV as whole-input Gaussian corruption. Use PJ's manifest-paired storage figure in the required DIP comparison. |
| Pending AP handoff | Exact raw model confusion matrix and prediction records require AP's original evaluation files or raw checkpoint. Epoch times are approximate values from the roadmap. |
