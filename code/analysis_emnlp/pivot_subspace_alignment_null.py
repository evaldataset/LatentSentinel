#!/usr/bin/env python
"""CHECK fix H-3 — Procrustes null distribution for the rank-3 subspace
alignment claim. The paper claims pairwise Procrustes disparity ≤0.04 across
3 backbones; a geometry-literate reviewer will object that 4 centroids in R^3
are nearly unconstrained under a 9-parameter Procrustes fit. We compute a
permutation null: for each pair of (model_a, model_b), draw N=10000 random
rank-3 bases (Gaussian -> QR-orthonormalized), project the 4 source centroids
from each model into the random basis, and compute Procrustes disparity. We
report the observed-disparity p-value vs the null.

Output: data/emnlp2026/pivot/subspace_alignment_null.json with
   { pair: { observed, null_mean, null_p05, null_p95, p_value, n_null } }

Reads centroids from `data/emnlp2026/pivot/subspace_alignment.json` (already on
disk, produced by `pivot_subspace_alignment.py`) and `hs_*.npz` caches for the
random-basis null (so the null sample IS conditional on each model's hidden-
state distribution, not pure-Gaussian — a stronger null).

CPU only, ~30 s with N=1000, ~5 min with N=10000.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import procrustes

REPO = Path(__file__).resolve().parents[2]
PIV = REPO / "data/emnlp2026/pivot"
BACKBONES = ["qwen_aligned", "llama_aligned", "mistral_aligned"]
CACHES = {b: f"hs_{b}_L{ {'qwen_aligned':20,'llama_aligned':16,'mistral_aligned':16}[b] }.npz"
          for b in BACKBONES}


def random_basis(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian then QR -> orthonormal d-by-k matrix."""
    G = rng.standard_normal((d, k))
    Q, _ = np.linalg.qr(G)
    return Q  # (d, k)


def centroids_in_basis(Hbs: np.ndarray, bs_ds: np.ndarray,
                       Hin_ben: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project 4-source centroids (indist + 3 bs sources) into a given basis."""
    sources = ["indist_benign"] + sorted(set(map(str, bs_ds)))
    cents = []
    cents.append(Hin_ben.mean(0) @ basis)
    bs_str = np.array(list(map(str, bs_ds)))
    for s in sources[1:]:
        mask = bs_str == s
        cents.append(Hbs[mask].mean(0) @ basis)
    return np.array(cents, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-null", type=int, default=1000,
                    help="random-basis samples per pair (default 1000)")
    a = ap.parse_args()

    # observed disparities (Procrustes-aligned centroids per the original script)
    obs_data = json.loads((PIV / "subspace_alignment.json").read_text())
    obs_disparity = {}
    cents_obs = {b: np.array(
        obs_data["per_model_configurations"][b]["centroids_in_register_subspace"],
        dtype=np.float64) for b in BACKBONES}
    for i, b1 in enumerate(BACKBONES):
        for b2 in BACKBONES[i+1:]:
            _, _, disp = procrustes(cents_obs[b1], cents_obs[b2])
            obs_disparity[f"{b1}__vs__{b2}"] = float(disp)

    # null: random rank-3 basis per model, project the same 4 centroids,
    # then Procrustes-compare. Centroids per model are computed from the
    # model's OWN cached Hin/Hbs (so the per-model marginal distribution is
    # respected, not just random Gaussian).
    rng = np.random.default_rng(20260521)
    data = {}
    for b in BACKBONES:
        z = np.load(PIV / CACHES[b], allow_pickle=True)
        ben = np.where(z["y"] == 0)[0]
        data[b] = dict(
            Hin_ben=z["Hin"][ben].astype(np.float32),
            Hbs=z["Hbs"].astype(np.float32),
            bs_ds=z["bs_ds"], d=z["Hin"].shape[1])

    out = {"n_null": int(a.n_null),
           "observed_disparity": obs_disparity,
           "per_pair": {}}
    for i, b1 in enumerate(BACKBONES):
        for b2 in BACKBONES[i+1:]:
            null = []
            for _ in range(a.n_null):
                B1 = random_basis(data[b1]["d"], 3, rng)
                B2 = random_basis(data[b2]["d"], 3, rng)
                c1 = centroids_in_basis(data[b1]["Hbs"], data[b1]["bs_ds"],
                                        data[b1]["Hin_ben"], B1)
                c2 = centroids_in_basis(data[b2]["Hbs"], data[b2]["bs_ds"],
                                        data[b2]["Hin_ben"], B2)
                _, _, disp = procrustes(c1, c2)
                null.append(float(disp))
            null = np.array(null)
            obs = obs_disparity[f"{b1}__vs__{b2}"]
            p = float((null <= obs).mean())
            out["per_pair"][f"{b1}__vs__{b2}"] = dict(
                observed=round(obs, 5),
                null_mean=round(float(null.mean()), 5),
                null_median=round(float(np.median(null)), 5),
                null_p05=round(float(np.quantile(null, 0.05)), 5),
                null_p95=round(float(np.quantile(null, 0.95)), 5),
                p_value_observed_le_null=round(p, 5))
            print(f"  {b1} vs {b2}: obs={obs:.4f}  "
                  f"null mean={null.mean():.4f} "
                  f"[{np.quantile(null,0.05):.4f},"
                  f"{np.quantile(null,0.95):.4f}]  "
                  f"p(null<=obs)={p:.4f}")
    op = PIV / "subspace_alignment_null.json"
    op.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
