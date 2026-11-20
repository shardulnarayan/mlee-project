# Validation record

Checked on 20 November 2026. No Git command was executed for this change.

## Completed checks

- All new Python files compiled successfully with Python 3.12.
- Eleven tests passed: five-class metrics, absent-class handling, minority-class
  errors, invalid inputs, sample SD, manifest integrity, pending-table handling,
  seed aggregation, blur, brightness clipping, JPEG and rotation behaviour.
- The canonical manifest contains 1,004 images with 702/151/151 assignments and
  no source-photo group crossing split boundaries.
- All manifest-selected raw and processed image files were present and nonempty.
- CLI help and existing-evidence table assembly ran successfully.
- Existing per-class, feature-ablation and grouped-CV CSVs were copied into the
  supplementary output directory with their labels and numeric values intact.

## Not executed

The full nine-model training, quantitative attention and CNN corruption runs
have not been executed in this session. The available bundled Python runtime
lacks torch, torchvision, OpenCV and matplotlib; the existing project virtual
environment's Python launcher failed to start. This is an environment limitation,
not a completed end-to-end ML validation. Activate a working project environment
and run the documented preflight and suite before treating the new code as the
source of measured CNN results.

Tables 6 and 7 remain explicitly pending in `table_status.json`. Measurement
outputs are populated from completed evaluation stages. Default attention masks
and corruption settings are documented in the supplementary README.
