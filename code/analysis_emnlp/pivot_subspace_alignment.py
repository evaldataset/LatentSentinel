#!/usr/bin/env python
"""B2 — cross-model alignment of the rank-3 register subspace.

Direct hidden-state dimensions differ across backbones (Qwen 3584 / Llama 4096 /
Mistral 4096), so we compare the *geometric configuration* the rank-3 register
subspace induces on the 4 benign sources (in-dist Alpaca-style + 3 diverse
sources XSTest / OR-Bench / ToxicChat). If the same 4-source shape appears in
all three models, that is strong evidence the register subspace encodes a
universal property of the data, not a model-specific quirk.

Protocol (CPU, cached states only):
  1. For each backbone, recompute the LEACE-rank-3 register basis Q_m
     (the column space of the projection (I - eraser) that LEACE removes).
  2. For each backbone, compute the 4 per-source mean hidden states m_s.
  3. Project (m_s - mu_m) onto Q_m -> a 4x3 configuration C_m.
  4. Procrustes-align C_qwen, C_llama, C_mistral (orthogonal rotation + scale
     + translation); report the Procrustes disparity (Frobenius residual /
     trace) and the pairwise canonical correlations of the column spaces.

Lower disparity / higher CCA <=> stronger universality of the register
subspace.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial import procrustes

REPO = Path(__file__).resolve().parents[2]
PIV = REPO / "data/emnlp2026/pivot"
CACHES = {"qwen_aligned": "hs_qwen_aligned_L20.npz",
          "llama_aligned": "hs_llama_aligned_L16.npz",
          "mistral_aligned": "hs_mistral_aligned_L16.npz"}


def fit_register_basis(Hin, y, Hbs, bs_ds):
    """Return (mu, Q) where Q is the 4-source rank-3 register basis (d x 3)."""
    ben = np.where(y == 0)[0]
    src = ["indist_benign"] + sorted(set(map(str, bs_ds)))
    Xfit = np.vstack([Hin[ben], Hbs])
    zlab = np.array([0] * len(ben) + [src.index(str(s)) for s in bs_ds])
    Zoh = np.eye(len(src))[zlab][:, 1:]
    mu = Xfit.mean(0); Xc = Xfit - mu; d = Xc.shape[1]
    eps = 1e-3
    Sig = (Xc.T @ Xc) / len(Xc) + eps * np.eye(d)
    w, U = np.linalg.eigh(Sig); w = np.clip(w, eps, None)
    isq = (U * w ** -0.5) @ U.T
    Zc = Zoh - Zoh.mean(0)
    M = isq @ ((Xc.T @ Zc) / len(Xc))
    Um, sm, _ = np.linalg.svd(M, full_matrices=False)
    r = max(int((sm > 1e-6).sum()), 1)
    return mu.astype(np.float32), Um[:, :r].astype(np.float32), r


def per_source_centroids(Hin, y, Hbs, bs_ds):
    """4 centroids: in-dist benign + 3 OOD sources."""
    ben = np.where(y == 0)[0]
    sources = sorted(set(map(str, bs_ds)))
    cents = {"indist_benign": Hin[ben].mean(0)}
    bs_str = np.array(list(map(str, bs_ds)))
    for s in sources:
        cents[s] = Hbs[bs_str == s].mean(0)
    # canonical order
    order = ["indist_benign"] + sources
    return order, np.vstack([cents[s] for s in order])  # (4, d)


def main():
    configs = {}
    for model, fn in CACHES.items():
        z = np.load(PIV / fn, allow_pickle=True)
        Hin, y, Hbs, bs_ds = z["Hin"], z["y"], z["Hbs"], z["bs_ds"]
        mu, Q, rnk = fit_register_basis(Hin, y, Hbs, bs_ds)
        order, C = per_source_centroids(Hin, y, Hbs, bs_ds)
        C_proj = (C - mu) @ Q                          # (4, rank)
        configs[model] = dict(order=order, C_proj=C_proj, rank=rnk,
                              Q_shape=list(Q.shape))
        print(f"  {model}: rank={rnk}, 4 source centroids projected onto "
              f"register basis ({C_proj.shape})")

    # Procrustes pairwise
    out = {"per_model_configurations":
           {m: {"source_order": configs[m]["order"],
                "rank": configs[m]["rank"],
                "centroids_in_register_subspace": configs[m]["C_proj"].tolist()}
            for m in configs},
           "pairwise_alignment": {}}
    models = list(configs)
    print("\n=== Procrustes pairwise alignment (4 source centroids in rank-3 register) ===")
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            A = configs[models[i]]["C_proj"]
            B = configs[models[j]]["C_proj"]
            # align to common dimensionality (min rank)
            r = min(A.shape[1], B.shape[1])
            A_, B_, disparity = procrustes(A[:, :r], B[:, :r])
            # interpret: disparity in [0, 1]; lower = more aligned
            out["pairwise_alignment"][f"{models[i]}__vs__{models[j]}"] = {
                "rank_used": int(r),
                "procrustes_disparity": round(float(disparity), 4),
                "interpretation": ("strongly aligned (universal)" if disparity < 0.1
                                   else "moderately aligned" if disparity < 0.3
                                   else "weakly aligned (model-specific)")}
            print(f"  {models[i]} <-> {models[j]}: disparity={disparity:.4f} "
                  f"({out['pairwise_alignment'][f'{models[i]}__vs__{models[j]}']['interpretation']})")
    op = PIV / "subspace_alignment.json"
    op.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
