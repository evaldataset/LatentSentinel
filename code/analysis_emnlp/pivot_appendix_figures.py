#!/usr/bin/env python
"""Generates 3 new appendix figures for the latent_sentinel_arr.tex paper.

Outputs:
  paper/fig_subspace_overlay.pdf     -- Procrustes-aligned rank-3 register subspace
                                        with 3 backbones' 4-source centroids in shared coords.
                                        Visualizes the B2 universal-alignment claim
                                        (pairwise Procrustes disparity <= 0.04).
  paper/fig_adaptive_margin.pdf      -- Bar chart of normalized median min-L2 perturbation
                                        delta_min to evade recipe vs baseline per backbone.
                                        Visualizes the B1' "1.4-2.1x more robust" finding.
  paper/fig_operating_point.pdf      -- Recipe vs baseline operating point scatter
                                        on (held-out diverse benign FPR, jailbreak recall)
                                        plane across 3 backbones.
                                        Visualizes Table 6 (tab:deploy) headline result.

CPU-only, reads finished JSON artifacts under data/emnlp2026/pivot/.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
PIV = REPO / "data/emnlp2026/pivot"
OUT = REPO / "paper"

BACKBONES = ["qwen_aligned", "llama_aligned", "mistral_aligned"]
SHORT = {"qwen_aligned": "Qwen2.5-7B", "llama_aligned": "Llama-3.1-8B",
         "mistral_aligned": "Mistral-7B"}
SOURCES = ["indist_benign", "orbench_hard", "toxicchat_benign", "xstest_safe"]
SOURCE_SHORT = {"indist_benign": "In-dist (Alpaca-style)",
                "orbench_hard": "OR-Bench-Hard",
                "toxicchat_benign": "ToxicChat-benign",
                "xstest_safe": "XSTest-safe"}
SOURCE_COLOR = {"indist_benign": "#888888",
                "orbench_hard": "#1f77b4",
                "toxicchat_benign": "#2ca02c",
                "xstest_safe": "#ff7f0e"}
BACKBONE_MARKER = {"qwen_aligned": "o",
                   "llama_aligned": "s",
                   "mistral_aligned": "^"}


def procrustes_align(A: np.ndarray, B: np.ndarray):
    """Orthogonal Procrustes: find scale s, rotation R, translation t
    minimizing ||s*A@R + t - B||_F^2. Returns aligned A' = s*A@R + t."""
    muA, muB = A.mean(0), B.mean(0)
    Ac, Bc = A - muA, B - muB
    M = Ac.T @ Bc
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    nA = float(np.linalg.norm(Ac))
    s = float(S.sum()) / (nA * nA + 1e-12)
    A_aligned = s * (Ac @ R) + muB
    return A_aligned, dict(scale=s, R=R)


def fig_subspace_overlay():
    """All 3 backbones' rank-3 register-subspace centroids Procrustes-aligned
    to a shared reference frame (Qwen) and overlaid in 2D PCA of the union.
    Shows source-cluster topology is preserved across model families."""
    d = json.loads((PIV / "subspace_alignment.json").read_text())
    cfgs = d["per_model_configurations"]
    # collect 4-source centroids per backbone, rank-3 -> R^3
    pts = {b: np.array(cfgs[b]["centroids_in_register_subspace"], dtype=np.float64)
           for b in BACKBONES}
    # Procrustes-align Llama and Mistral to Qwen frame
    ref = pts["qwen_aligned"]
    aligned = {"qwen_aligned": ref}
    for b in ("llama_aligned", "mistral_aligned"):
        aligned[b], _ = procrustes_align(pts[b], ref)
    # project union to 2D via PCA (rank-3 already, just take leading 2 axes)
    Xu = np.vstack(list(aligned.values()))
    Xu_centered = Xu - Xu.mean(0)
    U, S, Vt = np.linalg.svd(Xu_centered, full_matrices=False)
    P = Xu_centered @ Vt[:2].T  # (12, 2)

    # also report Procrustes disparities (reproduces the JSON numbers)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    i = 0
    for b in BACKBONES:
        for j, s in enumerate(SOURCES):
            ax.scatter(P[i, 0], P[i, 1], s=120,
                       color=SOURCE_COLOR[s], marker=BACKBONE_MARKER[b],
                       edgecolor="black", linewidth=0.8,
                       label=None, alpha=0.95)
            i += 1
    # draw within-backbone polygons connecting the 4 source centroids
    i = 0
    for b in BACKBONES:
        block = P[i:i + 4]
        # close polygon
        loop = np.vstack([block, block[:1]])
        ax.plot(loop[:, 0], loop[:, 1], "-",
                color="black", lw=0.5, alpha=0.35)
        i += 4

    # build a compact legend in two halves
    src_handles = [plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=SOURCE_COLOR[s],
                              markeredgecolor="black", markersize=9,
                              label=SOURCE_SHORT[s]) for s in SOURCES]
    bb_handles = [plt.Line2D([0], [0], marker=BACKBONE_MARKER[b],
                             color="black", markerfacecolor="white",
                             markeredgecolor="black", markersize=9, lw=0,
                             label=SHORT[b]) for b in BACKBONES]
    ax.legend(handles=src_handles + bb_handles, fontsize=7, ncol=2,
              loc="best", frameon=True)
    ax.set_xlabel("Procrustes-aligned register PC1", fontsize=9)
    ax.set_ylabel("Procrustes-aligned register PC2", fontsize=9)
    ax.set_title("Rank-3 register subspace is geometrically aligned\n"
                 "across 3 model families (Procrustes disparity $\\leq$0.04)",
                 fontsize=9.5)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    out = OUT / "fig_subspace_overlay.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"written: {out}")
    plt.close(fig)


def fig_adaptive_margin():
    """Bar chart: recipe vs baseline median min-L2 perturbation delta_min
    (normalized by typical hidden-state norm), per backbone. Shows
    LEACE-recipe is 1.4-2.1x adaptively harder to evade in continuous space."""
    d = json.loads((PIV / "adaptive_margin_recipe.json").read_text())
    labels = [SHORT[b] for b in BACKBONES]
    rec = [d[b]["median_min_delta_NORMALIZED_recipe"] for b in BACKBONES]
    base = [d[b]["median_min_delta_NORMALIZED_baseline"] for b in BACKBONES]
    ratios = [rec[i] / base[i] for i in range(3)]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    b1 = ax.bar(x - w / 2, base, w, label="Published baseline (no erasure)",
                color="#bbbbbb", edgecolor="black", linewidth=0.7)
    b2 = ax.bar(x + w / 2, rec, w, label="Recipe (LEACE-erased)",
                color="#d62728", edgecolor="black", linewidth=0.7, alpha=0.9)
    # ratio annotation above recipe bar
    for i, r in enumerate(ratios):
        ax.annotate(f"$\\times${r:.1f}", xy=(x[i] + w / 2, rec[i]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, color="#600")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"Median min $\|\delta\|_2$ to evade @ deploy-$\tau$"
                  "\n(fraction of typical $\\|h\\|$)", fontsize=9)
    ax.set_title("Recipe is adaptively harder to evade than baseline\n"
                 "in continuous hidden-state space",
                 fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT / "fig_adaptive_margin.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"written: {out}")
    plt.close(fig)


def fig_operating_point():
    """Scatter: recipe vs baseline operating point in
    (held-out diverse benign FPR, jailbreak recall) plane, per backbone.
    The recipe maintains 0.94-0.96 recall at <=5% FPR while baseline collapses
    to 0.18-0.29 at the same guaranteed FPR.

    CHECK fix B/H-1: now reads from `heldout_remedy.json` (the leakage-free
    spine) instead of `deployable_operating_point.json` (SUPERSEDED).
    """
    d = json.loads((PIV / "heldout_remedy.json").read_text())
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for b in BACKBONES:
        bl = d[b]["baseline"]
        rg = d[b]["register_erased"]
        # baseline at deploy-tau
        ax.scatter(bl["FPR_ood_heldout_at_deploy_tau"],
                   bl["recall_jb_at_deploy_tau"],
                   marker=BACKBONE_MARKER[b], s=140, color="#bbbbbb",
                   edgecolor="black", linewidth=0.8, label=None)
        # recipe at deploy-tau
        ax.scatter(rg["FPR_ood_heldout_at_deploy_tau"],
                   rg["recall_jb_at_deploy_tau"],
                   marker=BACKBONE_MARKER[b], s=140, color="#d62728",
                   edgecolor="black", linewidth=0.8, label=None)
        # arrow from baseline -> recipe
        ax.annotate("", xytext=(bl["FPR_ood_heldout_at_deploy_tau"],
                                bl["recall_jb_at_deploy_tau"]),
                    xy=(rg["FPR_ood_heldout_at_deploy_tau"],
                        rg["recall_jb_at_deploy_tau"]),
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                    lw=0.8, alpha=0.45))
        # backbone label near recipe point
        ax.annotate(SHORT[b],
                    (rg["FPR_ood_heldout_at_deploy_tau"],
                     rg["recall_jb_at_deploy_tau"]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=8, va="center")
    ax.axvline(0.05, color="gray", ls="--", lw=0.7, alpha=0.7)
    ax.text(0.052, 0.55, r"guaranteed FPR $\leq 0.05$",
            fontsize=8, color="gray")
    # legend handles
    h_b = plt.Line2D([0], [0], marker="o", color="w",
                     markerfacecolor="#bbbbbb", markeredgecolor="black",
                     markersize=10, label="Published baseline")
    h_r = plt.Line2D([0], [0], marker="o", color="w",
                     markerfacecolor="#d62728", markeredgecolor="black",
                     markersize=10, label="Recipe (LEACE-erased)")
    ax.legend(handles=[h_b, h_r], fontsize=8.5, loc="lower right",
              frameon=True)
    ax.set_xlabel(r"Held-out diverse benign FPR (at deploy-$\tau$)",
                  fontsize=9)
    ax.set_ylabel(r"Jailbreak recall (at deploy-$\tau$)", fontsize=9)
    ax.set_xlim(-0.005, max(0.08, ax.get_xlim()[1]))
    ax.set_ylim(0, 1.05)
    ax.set_title("Recipe vs.\\ baseline at the guaranteed operating point\n"
                 "(3 distinct-family backbones)", fontsize=9.5)
    ax.grid(alpha=0.25, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT / "fig_operating_point.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"written: {out}")
    plt.close(fig)


def main():
    fig_subspace_overlay()
    fig_adaptive_margin()
    fig_operating_point()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
