# Data

Raw survey data and the full injected lens dataset are **not included**
in this public repository due to size and licensing constraints.

## Expected directory structure

For full reproduction, the following directories should be populated:

```
artifacts/
    injected_dataset/
        commission_v2_flat/
            pos_00000.npz ... pos_00199.npz   (200 positive examples)
            neg_00000.npz ... neg_01999.npz   (2000 negative examples)
    groupkfold/
        gkf_unweighted_seed0/
            fold00/test_predictions.npz
            fold01/test_predictions.npz
            fold02/test_predictions.npz
            fold03/test_predictions.npz
            fold04/test_predictions.npz
            summary.json
            run_config.json
            locked_gradcam/
                fold00_model.pt ... fold04_model.pt
```

## File formats

Each `.npz` dataset file contains:
- `image`: 101x101 float32 array (single-channel galaxy cutout)
- `label`: int (0 = non-lens, 1 = lens candidate)
- `params`: dict with keys including `ra`, `dec`

Each `test_predictions.npz` contains:
- `idx_test`: dataset indices for this fold's test set
- `y`: true labels
- `yhat`: predicted labels (binary)
- `scores`: raw logit scores
- `thr`: calibrated threshold
- `groups_test`: background group IDs

## Dataset summary

| Property | Value |
|---|---|
| Total images | 2,200 |
| Positives | 200 (simulated arc injection) |
| Negatives | 2,000 (real galaxy cutouts) |
| Background groups | 217 unique parent galaxies |
| Image size | 101 x 101 pixels, single channel |

## Demo data

A small example FITS cutout is included at `demo/example_cutout.fits`
for testing the inference pipeline. See `demo/README.md`.

## Data availability

Contact the authors for access to the full dataset and locked evaluation
artifacts.
