#!/usr/bin/env python3
"""
01_fp_concentration.py

Compute Gini coefficient, Lorenz curve, rank-ordered FP counts, and
permutation null for the distribution of false positives across
background groups in the canonical 5-fold evaluation.

Inputs:
    artifacts/groupkfold/gkf_unweighted_seed0/fold{00..04}/test_predictions.npz

Outputs:
    results/fp_concentration_stats.txt
    figures/fp_concentration_twopanel.pdf
    figures/gini_permutation_null.pdf
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import gini, lorenz

# ── Configuration ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "artifacts/groupkfold/gkf_unweighted_seed0"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
N_FOLDS = 5
N_PERMUTATIONS = 2000
RNG_SEED = 42


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load predictions ───────────────────────────────────────────────
    print(f"[1] Loading {N_FOLDS} folds from {EXP_DIR}")
    group_fp = defaultdict(int)
    group_neg = defaultdict(int)

    for fold in range(N_FOLDS):
        npz_path = EXP_DIR / f"fold{fold:02d}/test_predictions.npz"
        if not npz_path.exists():
            sys.exit(f"[ERROR] Missing: {npz_path}\n"
                     f"  Place locked prediction artifacts under {EXP_DIR}/")
        z = np.load(npz_path)
        y = z["y"].astype(int)
        yhat = z["yhat"].astype(int)
        groups = z["groups_test"]
        for i in range(len(y)):
            if y[i] == 0:
                group_neg[groups[i]] += 1
                if yhat[i] == 1:
                    group_fp[groups[i]] += 1

    # ── 2. Per-group statistics ───────────────────────────────────────────
    all_groups = sorted(group_neg.keys())
    fp_counts = np.array([group_fp.get(g, 0) for g in all_groups])
    neg_counts = np.array([group_neg[g] for g in all_groups])
    fpr_per_group = fp_counts / neg_counts
    n_groups = len(fp_counts)
    total_fp = int(fp_counts.sum())

    G_count = gini(fp_counts)
    G_rate = gini(fpr_per_group)

    sorted_fp = np.sort(fp_counts)
    n_top10 = max(1, int(np.ceil(0.10 * n_groups)))
    fp_share_top10 = sorted_fp[-n_top10:].sum() / sorted_fp.sum()

    desc_fp = np.sort(fp_counts)[::-1]
    cum_desc = np.cumsum(desc_fp) / desc_fp.sum()
    idx_90 = int(np.searchsorted(cum_desc, 0.90)) + 1

    n_catastrophic = int((fpr_per_group == 1.0).sum())
    n_clean = int((fpr_per_group == 0.0).sum())

    print(f"    Groups: {n_groups}  FPs: {total_fp}")
    print(f"    Gini (count): {G_count:.3f}  Gini (rate): {G_rate:.3f}")
    print(f"    Top 10% share: {fp_share_top10*100:.0f}%  Groups for 90%: {idx_90}")
    print(f"    Catastrophic: {n_catastrophic}  Clean: {n_clean}")

    # ── 3. Save stats ─────────────────────────────────────────────────────
    stats_path = RESULTS_DIR / "fp_concentration_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"Background groups: {n_groups}\n")
        f.write(f"Total false positives: {total_fp}\n")
        f.write(f"Gini coefficient (FP count): {G_count:.3f}\n")
        f.write(f"Gini coefficient (FPR per group): {G_rate:.3f}\n")
        f.write(f"Top 10% share of FPs: {fp_share_top10*100:.1f}%\n")
        f.write(f"Groups for 90% of FPs: {idx_90}\n")
        f.write(f"Catastrophic groups (FPR=1.0): {n_catastrophic}\n")
        f.write(f"Clean groups (FPR=0.0): {n_clean}\n")
    print(f"[2] Saved {stats_path}")

    # ── 4. Permutation null ───────────────────────────────────────────────
    print(f"[3] Running {N_PERMUTATIONS} permutations ...")
    rng = np.random.default_rng(RNG_SEED)
    null_ginis = np.empty(N_PERMUTATIONS)
    uniform_probs = np.ones(n_groups) / n_groups
    for i in range(N_PERMUTATIONS):
        sim = rng.multinomial(total_fp, uniform_probs)
        null_ginis[i] = gini(sim)

    p_value = (null_ginis >= G_count).sum() / N_PERMUTATIONS
    print(f"    Observed G={G_count:.3f}  p={p_value}")

    # ── 5. Two-panel figure ───────────────────────────────────────────────
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    # Panel A: Lorenz curve
    ax = axes[0]
    cum_groups, cum_fp_curve = lorenz(fp_counts)
    ax.plot(cum_groups, cum_fp_curve, color="black", linewidth=2.5, label="Observed")
    ax.plot([0, 1], [0, 1], "--", linewidth=1.4, color="gray", label="Equality line")
    ax.set_xlabel("Cumulative fraction of background groups")
    ax.set_ylabel("Cumulative fraction of false positives")
    ax.set_title(f"Lorenz curve (Gini = {G_count:.3f})")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    ax.text(-0.14, 1.04, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

    # Inset: permutation null
    ax_in = ax.inset_axes([0.52, 0.30, 0.44, 0.32])
    ax_in.hist(null_ginis, bins=25, color="0.82", edgecolor="0.5", linewidth=0.5)
    ax_in.axvline(G_count, color="black", linewidth=1.5)
    ax_in.set_xlabel("Gini", fontsize=7, labelpad=1)
    ax_in.set_ylabel("Count", fontsize=7, labelpad=1)
    ax_in.tick_params(labelsize=6)
    ax_in.set_title("Permutation null", fontsize=7, pad=2)

    # Panel B: Rank-ordered FP counts
    ax = axes[1]
    ranks = np.arange(1, n_groups + 1)
    ax.plot(ranks, desc_fp, color="black", linewidth=2.0)
    ax.axvline(n_top10, linestyle="--", linewidth=1.4, color="gray")
    ax.axvline(idx_90, linestyle=":", linewidth=1.4, color="black")
    ax.set_xlabel("Background groups ranked by FP count")
    ax.set_ylabel("False positives per group")
    ax.set_title(f"Rank-ordered FP counts (N={n_groups}, total FPs={total_fp})")
    ax.grid(alpha=0.2)
    ax.text(-0.14, 1.04, "B", transform=ax.transAxes, fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fp_concentration_twopanel.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fp_concentration_twopanel.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[4] Saved figures/fp_concentration_twopanel.pdf")
    print("[DONE]")


if __name__ == "__main__":
    main()
