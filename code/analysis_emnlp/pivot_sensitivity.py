#!/usr/bin/env python
"""CHECK.md E3+E4+E6+E7 — paired test + latency + split-seed sensitivity +
LEACE rank sensitivity, all CPU re-analyses of the cached hs_*.npz. Reuses
the EXACT split + LEACE protocol of pivot_heldout_remedy.py.

E3  Paired Wilcoxon (LEACE vs reg-bal) on deploy-recall over 3 split-seeds x 3 backbones.
E4  Recipe overhead latency: timing er() and conformal lookup over a 1k batch.
E6  Split-seed sensitivity of the held-out remedy numbers (5 seeds).
E7  LEACE rank sensitivity (rank 1, 2, 3, 5) at the canonical split-seed.

Env: pinned in requirements.txt (see `bootstrap-latents` conda env)
  python pivot_sensitivity.py
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
PIV = REPO / "data/emnlp2026/pivot"
CACHES = {"qwen_aligned": "hs_qwen_aligned_L20.npz",
          "llama_aligned": "hs_llama_aligned_L16.npz",
          "mistral_aligned": "hs_mistral_aligned_L16.npz"}
ALPHA = 0.05


def leace(Xfit, Zfit, eps=1e-3, rank_override=None):
    mu = Xfit.mean(0); Xc = Xfit - mu; d = Xc.shape[1]
    Sig = (Xc.T @ Xc) / len(Xc) + eps * np.eye(d)
    w, U = np.linalg.eigh(Sig); w = np.clip(w, eps, None)
    isq = (U * w ** -0.5) @ U.T; sq = (U * w ** 0.5) @ U.T
    Zc = Zfit - Zfit.mean(0)
    M = isq @ ((Xc.T @ Zc) / len(Xc))
    Um, sm, _ = np.linalg.svd(M, full_matrices=False)
    r = max(int((sm > 1e-6).sum()), 1)
    if rank_override is not None:
        r = min(rank_override, Um.shape[1])
    Q = Um[:, :r]
    R = sq @ (np.eye(d) - Q @ Q.T) @ isq
    return mu.astype(np.float32), R.astype(np.float32), int(r)


def conformal_tau(s, alpha=ALPHA):
    s = np.sort(s); k = int(np.ceil((len(s) + 1) * (1 - alpha))) - 1
    return float(s[min(k, len(s) - 1)])


def strat_3way(bs_ds, seed):
    rng = np.random.default_rng(seed); eraser, depcal, test = [], [], []
    for s in sorted(set(map(str, bs_ds))):
        idx = np.where(np.array(list(map(str, bs_ds))) == s)[0]
        rng.shuffle(idx)
        n = len(idx); a, b = n // 3, 2 * n // 3
        eraser += list(idx[:a]); depcal += list(idx[a:b]); test += list(idx[b:])
    return (np.array(sorted(eraser)), np.array(sorted(depcal)),
            np.array(sorted(test)))


def train_probe(Htr, ytr):
    sc = StandardScaler().fit(Htr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Htr), ytr)
    return lambda X: clf.predict_proba(sc.transform(X))[:, 1]


def evaluate(score, Hin, y, te, Hbs, bs_test, bs_depcal):
    s_te = score(Hin[te]); yte = y[te]
    tau_dep = conformal_tau(score(Hbs[bs_depcal]))
    fpr = float((score(Hbs[bs_test]) >= tau_dep).mean())
    rec = float((s_te[yte == 1] >= tau_dep).mean())
    fpr_idcal = float((score(Hbs[bs_test])
                       >= conformal_tau(s_te[yte == 0])).mean())
    return dict(deploy_FPR=round(fpr, 4), deploy_recall=round(rec, 4),
                indistcal_FPR_heldout=round(fpr_idcal, 4),
                _s_jb=s_te[yte == 1], _tau=tau_dep)


def one_run(Hin, y, Hbs, bs_ds, seed, rank_override=None):
    # CHECK fix H-2: in-dist split now seeded by the outer `seed` arg, not
    # hardcoded 42. Reported E6 split-seed std now reflects true variance
    # across both the in-dist and bs partitions.
    rng = np.random.default_rng(seed)
    ben = np.where(y == 0)[0]; jb = np.where(y == 1)[0]
    rng.shuffle(ben); rng.shuffle(jb)
    cal_b, te_b = ben[:len(ben) // 2], ben[len(ben) // 2:]
    tr = np.array(sorted(list(cal_b) + list(jb[:len(jb) // 2])))
    te = np.array(sorted(list(te_b) + list(jb[len(jb) // 2:])))
    bs_er, bs_dc, bs_te = strat_3way(bs_ds, seed)
    src = ["indist_benign"] + sorted(set(map(str, bs_ds[bs_er])))
    # CHECK fix H-1: LEACE fit uses Hin[cal_b] only (in-dist test te_b excluded
    # from eraser-fit data; eraser is unsupervised w.r.t. y but mu/Sigma must
    # not be computed on the in-dist test partition).
    Xfit = np.vstack([Hin[cal_b], Hbs[bs_er]])
    zlab = np.array([0] * len(cal_b) +
                    [src.index(str(s)) for s in bs_ds[bs_er]])
    Zoh = np.eye(len(src))[zlab][:, 1:]
    mu, R, rnk = leace(Xfit, Zoh, rank_override=rank_override)
    er = lambda X: ((X - mu) @ R.T + mu).astype(np.float32)
    base = evaluate(train_probe(Hin[tr], y[tr]), Hin, y, te,
                    Hbs, bs_te, bs_dc)
    ers = evaluate(train_probe(er(Hin[tr]), y[tr]),
                   er(Hin), y, te, er(Hbs), bs_te, bs_dc)
    Htr2 = np.vstack([Hin[tr], Hbs[bs_er]])
    ytr2 = np.concatenate([y[tr], np.zeros(len(bs_er), int)])
    rbt = evaluate(train_probe(Htr2, ytr2), Hin, y, te,
                   Hbs, bs_te, bs_dc)
    return dict(rank=rnk,
                baseline=base, register_erased=ers,
                register_balanced=rbt, er=er, mu=mu, R=R)


def main():
    out = {"E3_LEACE_vs_regbal_paired": {},
           "E4_latency_us_per_prompt": {},
           "E6_split_seed_sensitivity": {},
           "E7_LEACE_rank_sensitivity": {}}
    seeds = [20260520, 20260521, 20260522, 20260523, 20260524]
    leace_deploy_recalls, regbal_deploy_recalls = [], []
    # ---- E6 (sensitivity) + collect paired data for E3 ----
    for model, fn in CACHES.items():
        z = np.load(PIV / fn, allow_pickle=True)
        Hin, y, Hbs, bs_ds = z["Hin"], z["y"], z["Hbs"], z["bs_ds"]
        per_seed = []
        for sd in seeds:
            r = one_run(Hin, y, Hbs, bs_ds, sd)
            per_seed.append({
                "seed": sd, "rank": r["rank"],
                "erased_deploy_recall": r["register_erased"]["deploy_recall"],
                "erased_deploy_FPR":    r["register_erased"]["deploy_FPR"],
                "erased_indistcal_FPR_heldout":
                    r["register_erased"]["indistcal_FPR_heldout"],
                "regbal_deploy_recall": r["register_balanced"]["deploy_recall"],
                "regbal_deploy_FPR":    r["register_balanced"]["deploy_FPR"]})
            leace_deploy_recalls.append(r["register_erased"]["deploy_recall"])
            regbal_deploy_recalls.append(r["register_balanced"]["deploy_recall"])
        recs = np.array([s["erased_deploy_recall"] for s in per_seed])
        out["E6_split_seed_sensitivity"][model] = {
            "n_seeds": len(seeds),
            "erased_deploy_recall_mean": round(float(recs.mean()), 4),
            "erased_deploy_recall_std":  round(float(recs.std()),  4),
            "per_seed": per_seed}
        print(f"\n[E6] {model} erased deploy-recall over {len(seeds)} split-seeds: "
              f"{recs.mean():.4f} +/- {recs.std():.4f}  (per seed {list(recs)})")

    # ---- E3 paired Wilcoxon ----
    le, rb = np.array(leace_deploy_recalls), np.array(regbal_deploy_recalls)
    diff = rb - le
    try:
        stat, p = wilcoxon(le, rb, alternative="two-sided",
                           zero_method="pratt", correction=False)
    except ValueError as e:
        stat, p = float("nan"), 1.0; print("[E3] wilcoxon failed:", e)
    out["E3_LEACE_vs_regbal_paired"] = {
        "n_pairs": int(len(le)),
        "LEACE_deploy_recall_mean": round(float(le.mean()), 4),
        "regbal_deploy_recall_mean": round(float(rb.mean()), 4),
        "delta_mean":  round(float(diff.mean()), 4),
        "delta_min":   round(float(diff.min()),  4),
        "delta_max":   round(float(diff.max()),  4),
        "wilcoxon_W":  round(float(stat), 4),
        "wilcoxon_p":  float(p),
        "interpretation": ("reg-bal dominantly higher" if diff.mean() > 0
                           else "comparable / LEACE higher")}
    print(f"\n[E3] LEACE vs reg-bal: mean {le.mean():.4f} vs {rb.mean():.4f} "
          f"(delta {diff.mean():+.4f}, range [{diff.min():+.4f},{diff.max():+.4f}]); "
          f"Wilcoxon W={stat:.3f} p={p:.4f}")

    # ---- E7 LEACE rank sensitivity (Qwen only as representative) ----
    z = np.load(PIV / CACHES["qwen_aligned"], allow_pickle=True)
    Hin, y, Hbs, bs_ds = z["Hin"], z["y"], z["Hbs"], z["bs_ds"]
    for rk in [1, 2, 3, 5, 8]:
        r = one_run(Hin, y, Hbs, bs_ds, seed=seeds[0], rank_override=rk)
        out["E7_LEACE_rank_sensitivity"][f"rank_{rk}"] = {
            "actual_rank": r["rank"],
            "erased_deploy_recall": r["register_erased"]["deploy_recall"],
            "erased_deploy_FPR":    r["register_erased"]["deploy_FPR"],
            "erased_indistcal_FPR_heldout":
                r["register_erased"]["indistcal_FPR_heldout"]}
        print(f"[E7] Qwen LEACE rank {rk} (actual {r['rank']}): deploy-recall "
              f"{r['register_erased']['deploy_recall']:.4f}, "
              f"indist-cal heldout-FPR {r['register_erased']['indistcal_FPR_heldout']:.4f}")

    # ---- E4 latency ----
    # representative: Qwen er() projection + conformal lookup on a 1024-prompt batch
    d = Hin.shape[1]
    x = Hin[:1024]
    r_qwen = one_run(Hin, y, Hbs, bs_ds, seed=seeds[0])
    mu, R = r_qwen["mu"], r_qwen["R"]
    cal_scores = np.random.default_rng(0).random(1000)
    # warm-up
    _ = ((x - mu) @ R.T + mu); _ = np.searchsorted(np.sort(cal_scores), 0.5)
    # er() timing
    t0 = time.perf_counter()
    for _ in range(20):
        _ = ((x - mu) @ R.T + mu)
    er_us = (time.perf_counter() - t0) / (20 * len(x)) * 1e6
    # conformal lookup timing (per scalar)
    s_sorted = np.sort(cal_scores)
    t0 = time.perf_counter()
    for _ in range(200):
        _ = np.searchsorted(s_sorted, 0.5)
    cl_us = (time.perf_counter() - t0) / 200 * 1e6
    out["E4_latency_us_per_prompt"] = {
        "backbone_hidden_dim": int(d),
        "leace_projection_us_per_prompt": round(er_us, 3),
        "conformal_lookup_us_per_call":   round(cl_us, 3),
        "note": "Recipe overhead = ONE d x d projection + ONE quantile lookup; "
                "additive to the backbone forward (which dominates by orders "
                "of magnitude)."}
    print(f"\n[E4] Qwen recipe overhead: LEACE projection "
          f"{er_us:.2f} us/prompt; conformal lookup {cl_us:.2f} us/call.")

    op = PIV / "sensitivity.json"
    # strip non-serializable
    for m in out["E6_split_seed_sensitivity"].values():
        for s in m["per_seed"]: pass
    op.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
