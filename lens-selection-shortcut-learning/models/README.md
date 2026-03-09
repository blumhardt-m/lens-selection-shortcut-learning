# Model checkpoints

Locked fold checkpoints are **not included** in this public repository
due to file size constraints. They are available separately upon request
or via the project's data archive.

## Expected files

For full reproduction of the embedding and Grad-CAM analyses, place the
following checkpoints in this directory:

```
models/
    fold00_model.pt
    fold01_model.pt
    fold02_model.pt
    fold03_model.pt
    fold04_model.pt
```

Each file is a PyTorch `state_dict` for the ProtocolCNN architecture
(360,033 parameters, ~1.4 MB per file).

## Architecture

The model architecture is defined in `src/model.py` and is **fixed**
across all experiments:

- 4 conv blocks: 16 -> 32 -> 64 -> 128 channels
- Each block: Conv2d(3x3) + BatchNorm + ReLU + Pool
- 128-dim penultimate embedding (Linear(2048, 128) + ReLU)
- Linear(128, 1) sigmoid head

## Usage

```python
from src.model import build_model, load_checkpoint

model = build_model()
load_checkpoint(model, "models/fold00_model.pt")
```

## Provenance

Checkpoints were produced by retraining on the same deterministic fold
splits (seed=0, k=5) used for the manuscript evaluation. Group labels
and out-of-fold assignments are taken from the locked `test_predictions.npz`
files. See the provenance files in the analysis output directories for
SHA-256 hashes.
