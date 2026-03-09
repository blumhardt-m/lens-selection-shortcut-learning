#!/usr/bin/env bash
set -euo pipefail

# make_submission_figures.sh
# Regenerate key paper figures from locked evaluation artifacts.
# Requires: dataset + fold checkpoints + test_predictions.npz
# See data/README.md and models/README.md for setup.

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

mkdir -p figures results

echo "============================================================"
echo "  Generating submission figures"
echo "============================================================"

run_analysis() {
    local script="$1"
    local label="$2"
    echo ""
    echo "--- ${label} ---"
    if python "$script" 2>&1; then
        echo "[OK] ${label}"
    else
        echo "[SKIP] ${label} (missing data or checkpoint; see README)"
    fi
}

run_analysis analysis/01_fp_concentration.py "FP concentration (Gini / Lorenz)"
run_analysis analysis/02_gradcam_examples.py "Grad-CAM qualitative examples"
run_analysis analysis/03_gradcam_quantitative.py "Grad-CAM quantitative summary"
run_analysis analysis/04_embedding_diagnostics.py "Embedding-space diagnostics"

echo ""
echo "============================================================"
echo "  Output summary"
echo "============================================================"
echo ""
echo "Figures:"
ls -1 figures/*.{pdf,png} 2>/dev/null || echo "  (none generated)"
echo ""
echo "Results:"
ls -1 results/*.{txt,tsv} 2>/dev/null || echo "  (none generated)"
echo ""
echo "Done."
