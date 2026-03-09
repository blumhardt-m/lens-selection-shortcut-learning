# Demo: single-cutout inference

Run the ProtocolCNN lens classifier on a single galaxy cutout.

## Quick start

```bash
python demo/demo_inference.py
```

## Requirements

1. **FITS cutout** (101x101 pixels, single channel).
   Place at `demo/example_cutout.fits` or specify with `--fits`.

2. **Model checkpoint** (ProtocolCNN `state_dict`).
   Place at `models/fold00_model.pt` or specify with `--checkpoint`.

## Using your own data

```bash
python demo/demo_inference.py \
    --fits /path/to/your/cutout.fits \
    --checkpoint /path/to/your/model.pt
```

The script will:
- Load and normalize the FITS image (clip [p1, p99], subtract median, divide by range)
- Run a forward pass through the ProtocolCNN
- Print the positive-class logit and sigmoid probability

## Notes

- The model expects 101x101 pixel single-channel cutouts
- Normalization follows the same procedure used during training
- If the FITS file has multiple extensions, the first (HDU 0) is used
- If the checkpoint file is missing, the script exits with a clear error message
