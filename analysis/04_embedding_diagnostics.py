#!/usr/bin/env python3
"""
04_embedding_diagnostics.py

Test whether catastrophic and clean background groups separate in the
model's learned penultimate-layer embedding (128-dim).

Uses locked fold checkpoints and locked out-of-fold predictions.
Group labels come from test_predictions.npz (locked).

Outputs:
    results/embedding_separation_summary.txt
    results/embedding_distance_tests.tsv
    results/nearest_neighbor_purity.tsv
    figures/embedding_pca_per_group_logit.pdf
    figures/embedding_pca_per_group.pdf
    figures/embedding_tsne_per_group.pdf
"""

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "artifacts/groupkfold/gkf_unweighted_seed0"
CKPT_DIR = EXP_DIR / "locked_gradcam"
DATASET_DIR = ROOT / "artifacts/injected_dataset/commission_v2_flat"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def check_prerequisites():
    """Verify required files exist."""
    missing = []
    for fold in range(5):
        npz = EXP_DIR / f"fold{fold:02d}/test_predictions.npz"
        if not npz.exists():
            missing.append(str(npz))
        ckpt = CKPT_DIR / f"fold{fold:02d}_model.pt"
        if not ckpt.exists():
            missing.append(str(ckpt))
    if not DATASET_DIR.exists():
        missing.append(str(DATASET_DIR))
    return missing


def main():
    missing = check_prerequisites()
    if missing:
        print("[ERROR] Missing required files:")
        for m in missing:
            print(f"  {m}")
        print("\nSee models/README.md and data/README.md for setup instructions.")
        sys.exit(1)

    import torch
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import mannwhitneyu
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score, silhouette_score
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    sys.path.insert(0, str(ROOT))
    from src.model import build_model, load_checkpoint
    from src.dataset import load_dataset, bg_group_from_params

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────
    print("[1/6] Loading dataset ...")
    images, labels, params = load_dataset(DATASET_DIR)
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(bg_group_from_params(params))
    print(f"       n={len(labels)}  pos={labels.sum()}  neg={(labels==0).sum()}")

    # ── Build group labels from locked predictions ────────────────────────
    print("[2/6] Reading locked predictions ...")
    group_fp = defaultdict(int)
    group_neg_count = defaultdict(int)
    group_fold = {}
    fold_neg_indices = defaultdict(list)
    fold_neg_groups = defaultdict(list)

    for fold in range(5):
        z = np.load(EXP_DIR / f"fold{fold:02d}/test_predictions.npz")
        yf = z["y"].astype(int)
        yhatf = z["yhat"].astype(int)
        idxf = z["idx_test"]
        gf = z["groups_test"]
        for i in range(len(yf)):
            g = int(gf[i])
            group_fold[g] = fold
            if yf[i] == 0:
                group_neg_count[g] += 1
                fold_neg_indices[fold].append(int(idxf[i]))
                fold_neg_groups[fold].append(g)
                if yhatf[i] == 1:
                    group_fp[g] += 1

    group_fpr = {g: group_fp.get(g, 0) / group_neg_count[g] for g in group_neg_count}
    catastrophic_set = set(g for g, fpr in group_fpr.items() if fpr == 1.0)
    clean_all = sorted(g for g, fpr in group_fpr.items()
                       if fpr == 0.0 and group_neg_count[g] >= 3)
    rng_sel = np.random.default_rng(42)
    clean_selected_set = set(
        rng_sel.choice(clean_all, size=min(len(clean_all), len(catastrophic_set)),
                       replace=False)
    )
    print(f"       Catastrophic: {len(catastrophic_set)}  Clean: {len(clean_selected_set)}")

    # ── Load checkpoints and extract embeddings ───────────────────────────
    print("[3/6] Extracting embeddings ...")
    all_emb, all_logit, all_group, all_fold = [], [], [], []

    for fold in range(5):
        model = build_model()
        load_checkpoint(model, CKPT_DIR / f"fold{fold:02d}_model.pt")

        for i, didx in enumerate(fold_neg_indices[fold]):
            img = images[didx]
            x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                emb = model.embed(x).squeeze(0).numpy()
                logit = model(x).squeeze().item()
            all_emb.append(emb)
            all_logit.append(logit)
            all_group.append(fold_neg_groups[fold][i])
            all_fold.append(fold)

    all_emb = np.array(all_emb, dtype=np.float32)
    all_logit = np.array(all_logit, dtype=np.float32)
    all_group = np.array(all_group, dtype=np.int64)
    all_cat = np.array([1 if g in catastrophic_set else 0 for g in all_group])

    # ── Group-level aggregation (analysis groups only) ────────────────────
    print("[4/6] Aggregating by group ...")
    analysis_set = catastrophic_set | clean_selected_set
    grp_keys = sorted(analysis_set)
    grp_emb = []
    grp_cat = []
    grp_logit = []
    grp_fpr = []

    for g in grp_keys:
        mask = all_group == g
        grp_emb.append(all_emb[mask].mean(axis=0))
        grp_cat.append(1 if g in catastrophic_set else 0)
        grp_logit.append(float(all_logit[mask].mean()))
        grp_fpr.append(group_fpr[g])

    grp_emb = np.array(grp_emb, dtype=np.float32)
    grp_cat = np.array(grp_cat)
    grp_logit = np.array(grp_logit)
    grp_fpr = np.array(grp_fpr)

    n_cat = int(grp_cat.sum())
    n_cln = int((grp_cat == 0).sum())

    # ── Quantify separation ───────────────────────────────────────────────
    print("[5/6] Quantifying separation ...")
    scaler = StandardScaler()
    X = scaler.fit_transform(grp_emb)

    # LOO logistic regression
    loo = LeaveOneOut()
    loo_probs = np.zeros(len(grp_cat))
    for tr, te in loo.split(X):
        lr = LogisticRegressionCV(cv=3, max_iter=5000, random_state=42,
                                  solver="lbfgs", penalty="l2")
        lr.fit(X[tr], grp_cat[tr])
        loo_probs[te] = lr.predict_proba(X[te])[:, 1]
    loo_auc = roc_auc_score(grp_cat, loo_probs)

    # Permutation null (200 x 5-fold for speed)
    n_perm = 200
    rng_perm = np.random.default_rng(42)
    null_aucs = np.zeros(n_perm)
    for pi in range(n_perm):
        y_perm = rng_perm.permutation(grp_cat)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pp = np.zeros(len(y_perm))
        for tr, te in skf.split(X, y_perm):
            lr_p = LogisticRegressionCV(cv=3, max_iter=5000, random_state=42,
                                        solver="lbfgs", penalty="l2")
            lr_p.fit(X[tr], y_perm[tr])
            pp[te] = lr_p.predict_proba(X[te])[:, 1]
        null_aucs[pi] = roc_auc_score(y_perm, pp)
    p_auc = float((null_aucs >= loo_auc).sum() / n_perm)

    # Distance-based
    D = squareform(pdist(grp_emb, metric="euclidean"))
    cat_idx = np.where(grp_cat == 1)[0]
    cln_idx = np.where(grp_cat == 0)[0]
    sil = silhouette_score(grp_emb, grp_cat, metric="euclidean")

    # Nearest-neighbor purity
    nn_results = {}
    for k in [5, 10]:
        purities = []
        for i in range(len(grp_cat)):
            dists = D[i].copy()
            dists[i] = np.inf
            nn = np.argsort(dists)[:k]
            purities.append(float((grp_cat[nn] == grp_cat[i]).sum()) / k)
        nn_results[k] = float(np.mean(purities))

    # PCA
    pca = PCA(n_components=min(20, X.shape[1]))
    pc = pca.fit_transform(X)

    print(f"       LOO AUC = {loo_auc:.3f}  (perm p = {p_auc:.4f})")
    print(f"       Silhouette = {sil:.4f}")
    print(f"       NN purity k=5: {nn_results[5]:.3f}  k=10: {nn_results[10]:.3f}")

    # ── Save results ──────────────────────────────────────────────────────
    with open(RESULTS_DIR / "embedding_separation_summary.txt", "w") as f:
        f.write(f"LOO AUC: {loo_auc:.3f}  (perm p = {p_auc:.4f})\n")
        f.write(f"Null AUC: {null_aucs.mean():.3f} +/- {null_aucs.std():.3f}\n")
        f.write(f"Silhouette: {sil:.4f}\n")
        f.write(f"NN purity k=5: {nn_results[5]:.3f}\n")
        f.write(f"NN purity k=10: {nn_results[10]:.3f}\n")
        f.write(f"Groups: {n_cat} catastrophic, {n_cln} clean\n")

    # ── Figures ───────────────────────────────────────────────────────────
    print("[6/6] Generating figures ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev1 = pca.explained_variance_ratio_[0] * 100
    ev2 = pca.explained_variance_ratio_[1] * 100

    # PCA per-group, binary coloring
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pc[grp_cat == 0, 0], pc[grp_cat == 0, 1], s=40, alpha=0.7,
               c="white", edgecolors="black", linewidths=0.5, label="Clean (FPR=0)")
    ax.scatter(pc[grp_cat == 1, 0], pc[grp_cat == 1, 1], s=40, alpha=0.7,
               c="0.55", edgecolors="black", linewidths=0.5, label="Catastrophic (FPR=1)")
    ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_title(f"PCA -- per-group mean embeddings\nPC1 ({ev1:.1f}%) vs PC2 ({ev2:.1f}%)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "embedding_pca_per_group.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "embedding_pca_per_group.png", dpi=300, bbox_inches="tight")
    plt.close()

    # PCA per-group, logit coloring
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(pc[:, 0], pc[:, 1], s=45, alpha=0.75, c=grp_logit,
                    cmap="coolwarm", edgecolors="black", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="Mean logit", shrink=0.8)
    ax.set_title(f"PCA -- per-group mean embeddings (color = mean logit)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "embedding_pca_per_group_logit.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "embedding_pca_per_group_logit.png", dpi=300, bbox_inches="tight")
    plt.close()

    # t-SNE per-group, FPR coloring
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(15, n_cat + n_cln - 1), max_iter=1000)
    tsne_2d = tsne.fit_transform(pc[:, :20])

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(tsne_2d[:, 0], tsne_2d[:, 1], s=45, alpha=0.75, c=grp_fpr,
                    cmap="RdYlGn_r", edgecolors="black", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="Group FPR", shrink=0.8)
    ax.set_title("t-SNE -- per-group mean embeddings (color = FPR)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "embedding_tsne_per_group_fpr.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "embedding_tsne_per_group_fpr.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"       Saved figures to {FIG_DIR}/")
    print("[DONE]")


if __name__ == "__main__":
    main()
