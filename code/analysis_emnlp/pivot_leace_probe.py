#!/usr/bin/env python
"""PIVOT spine experiment: does a jailbreak-INTENT signal survive register erasure?

Closed-form LEACE (Belrose et al. 2023, "LEAst-squares Concept Erasure") on cached
Qwen2.5-7B_JB last-token hidden states (29 layers x 3000 prompts;
data/inference_results/all_hidden_states_by_layer.pkl). Labels:
0=Benign, 1..8 = attack templates/sources (MR/SR/PE/AS/DAN/multilingual/AdvBench/
HarmfulGoal). Y = jailbreak(1) vs benign(0).

Per layer we compare a logistic Y-probe (5-fold CV ROC-AUC):
 * BASE: on raw H.
 * V2 (decisive, amnesic): fit the LEACE eraser using ONLY jailbreak samples with
   Z = attack-family one-hot (benign EXCLUDED, so the eraser never sees the Y
   split), apply to ALL H, re-probe Y. Removes within-jailbreak template/register
   directions without leaking Y. Collapse => benign-vs-jailbreak separation rode
   register/template axes (causal, stronger than a correlational shortcut claim).
   Survive => a register-independent intent component exists (basis of the remedy).
 * RAND: erase a random subspace of the SAME dimension (control: drop must be
   register-specific, not mere rank reduction).
 * Z-recoverability after V2 (sanity: family no longer linearly decodable).

Env: pinned in requirements.txt (see `bootstrap-latents` conda env)
"""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np
from numpy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

REPO = Path(__file__).resolve().parents[2]
HS = REPO / "data/inference_results/all_hidden_states_by_layer.pkl"
LAB = REPO / "data/inference_results/labels_for_plot_base.npy"
OUT = REPO / "data/emnlp2026/pivot"
RNG = np.random.default_rng(42)


def leace_eraser(X_fit, Z_fit, eps=1e-3):
    """Return (mu, R) s.t. erased = (X - mu) @ R.T + mu removes all linear info
    about Z (cross-cov(X_erased, Z) = 0). Eraser is FIT on (X_fit, Z_fit)."""
    mu = X_fit.mean(0)
    Xc = X_fit - mu
    d = Xc.shape[1]
    Sig = (Xc.T @ Xc) / len(Xc) + eps * np.eye(d)
    w, U = eigh(Sig)
    w = np.clip(w, eps, None)
    Sig_isq = (U * (w ** -0.5)) @ U.T          # Sigma^{-1/2}
    Sig_sq = (U * (w ** 0.5)) @ U.T            # Sigma^{ 1/2}
    Zc = Z_fit - Z_fit.mean(0)
    Sxz = (Xc.T @ Zc) / len(Xc)                # d x k
    M = Sig_isq @ Sxz                          # whitened cross-cov
    # orthonormal basis of col(M) via SVD
    Um, sm, _ = np.linalg.svd(M, full_matrices=False)
    r = int((sm > 1e-6).sum())
    Q = Um[:, :max(r, 1)]
    Pw = np.eye(d) - Q @ Q.T                    # project out concept in whitened space
    R = Sig_sq @ Pw @ Sig_isq                   # eraser in original space
    return mu.astype(np.float32), R.astype(np.float32), r


def cv_auc(X, y, seed=42):
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=4)[:, 1]
    return float(roc_auc_score(y, p))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    H = pickle.load(open(HS, "rb"))
    lab = np.load(LAB)
    Y = (lab != 0).astype(int)                          # jailbreak=1, benign=0
    jb = lab != 0
    fam = lab[jb]                                        # attack-family (1..8) jb-only
    K = np.unique(fam)
    Zf = np.eye(len(K))[np.searchsorted(K, fam)][:, 1:]  # one-hot, drop 1 col
    layers = sorted(H.keys(), key=int)
    print(f"n={len(Y)} (jb={Y.sum()}/ben={(1-Y).sum()})  layers={len(layers)}  "
          f"jb-only n={jb.sum()} families={list(K)}")

    rows = []
    for L in layers:
        X = np.asarray(H[L], dtype=np.float32)
        base = cv_auc(X, Y)
        # V2: eraser fit on jailbreak-only family identity (Y never seen by eraser)
        mu, R, rnk = leace_eraser(X[jb], Zf)
        Xer = (X - mu) @ R.T + mu
        v2 = cv_auc(Xer, Y)
        # Z-recoverability after erasure (should fall toward chance)
        famY = (lab[jb] != lab[jb][0])  # not meaningful; use multiclass acc proxy
        z_before = cv_auc(X[jb][:, :], (fam == K[0]).astype(int)) \
            if len(K) > 1 else 0.5
        z_after = cv_auc(Xer[jb], (fam == K[0]).astype(int)) if len(K) > 1 else 0.5
        # RAND control: erase a random subspace of equal rank
        Qr, _ = np.linalg.qr(RNG.standard_normal((X.shape[1], max(rnk, 1))))
        Xrand = X - (X - X.mean(0)) @ (Qr @ Qr.T)
        rand = cv_auc(Xrand, Y)
        rows.append(dict(layer=int(L), base=round(base, 4),
                         v2_register_erased=round(v2, 4),
                         rand_subspace_ctrl=round(rand, 4),
                         erased_rank=int(rnk),
                         drop_v2=round(base - v2, 4),
                         drop_rand=round(base - rand, 4),
                         Zrecover_before=round(z_before, 3),
                         Zrecover_after=round(z_after, 3)))
        print(f"L{int(L):2d} base={base:.3f}  reg-erased={v2:.3f} "
              f"(Δ{base-v2:+.3f}, rank{rnk})  rand-ctrl={rand:.3f} "
              f"(Δ{base-rand:+.3f})  Zrec {z_before:.2f}->{z_after:.2f}", flush=True)

    best = max(rows, key=lambda r: r["base"])
    res = {"model": "Qwen2.5-7B_JB (cached)", "n": int(len(Y)),
           "n_jb": int(Y.sum()), "n_benign": int((1 - Y).sum()),
           "design": "V2 amnesic: LEACE eraser fit on jailbreak-only attack-family "
                     "one-hot (Y never in Z); applied to all; re-probe Y.",
           "best_base_layer": best, "per_layer": rows}
    (OUT / "leace_spine.json").write_text(json.dumps(res, indent=1))
    # Verdict heuristic
    mid = [r for r in rows if 8 <= r["layer"] <= 24]
    md = float(np.mean([r["drop_v2"] for r in mid]))
    mr = float(np.mean([r["drop_rand"] for r in mid]))
    print("\n=== VERDICT (mid-layers 8-24 mean) ===")
    print(f"  register-erasure AUC drop = {md:.3f}   random-subspace drop = {mr:.3f}")
    print(f"  specificity (reg-drop minus rand-drop) = {md-mr:.3f}")
    if md > 0.15 and md - mr > 0.08:
        print("  -> OUTCOME 2: jailbreak signal COLLAPSES under register erasure "
              "=> the linear signal IS register (causal claim, beats Wang/Fomin).")
    else:
        print("  -> OUTCOME 1: signal SURVIVES register erasure => a "
              "register-independent intent direction exists (basis of the remedy).")
    print(f"written: {OUT}/leace_spine.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
