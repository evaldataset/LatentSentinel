#!/usr/bin/env python
"""CHECK fix M-7 / G13 — LOSO companion: OR-Bench-Hard INCLUDED in calibration.

The paper's LOSO table (Appendix C, `tab:loso`) measures the worst-case
conditional-bound failure: OR-Bench-Hard held OUT of calibration, giving
92-98% cross-source FPR — by design of the over-refusal benchmark. The
deployer's actually-used setting includes OR-Bench-H in calibration, where
the conformal guarantee should hold. We add the "included" complement here.

For each backbone:
  - eraser-fit set = in-dist benign(cal_b) + ALL 3 bs sources (eraser slice)
  - deploy-calib set = ALL 3 bs sources (depcal slice)  -- includes OR-Bench-H
  - test eval = held-out OR-Bench-H test slice (from the 3-way disjoint split)

Reports baseline & erased FPR + recall @ deploy-tau on each held-out source,
when OR-Bench-H is in calibration. This is the actually-deployed setting.

CPU only.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
PIV = REPO / "data/emnlp2026/pivot"
CACHES = {"qwen_aligned": "hs_qwen_aligned_L20.npz",
          "llama_aligned": "hs_llama_aligned_L16.npz",
          "mistral_aligned": "hs_mistral_aligned_L16.npz"}
ALPHA = 0.05


def leace(Xfit, Zfit, eps=1e-3):
    mu = Xfit.mean(0); Xc = Xfit - mu; d = Xc.shape[1]
    Sig = (Xc.T @ Xc) / len(Xc) + eps * np.eye(d)
    w, U = np.linalg.eigh(Sig); w = np.clip(w, eps, None)
    isq = (U * w ** -0.5) @ U.T; sq = (U * w ** 0.5) @ U.T
    Zc = Zfit - Zfit.mean(0)
    M = isq @ ((Xc.T @ Zc) / len(Xc))
    Um, sm, _ = np.linalg.svd(M, full_matrices=False)
    r = max(int((sm > 1e-6).sum()), 1)
    Q = Um[:, :r]
    R = sq @ (np.eye(d) - Q @ Q.T) @ isq
    return mu.astype(np.float32), R.astype(np.float32), int(r)


def conformal_tau(s, alpha=ALPHA):
    s = np.sort(s); k = int(np.ceil((len(s) + 1) * (1 - alpha))) - 1
    return float(s[min(k, len(s) - 1)])


def strat_3way(bs_ds, seed=20260520):
    """Source-stratified 3-way split (eraser/dep-cal/test) covering ALL sources."""
    rng = np.random.default_rng(seed); eraser, depcal, test = [], [], []
    for s in sorted(set(map(str, bs_ds))):
        idx = np.where(np.array(list(map(str, bs_ds))) == s)[0]
        rng.shuffle(idx); n = len(idx); a, b = n // 3, 2 * n // 3
        eraser += list(idx[:a]); depcal += list(idx[a:b]); test += list(idx[b:])
    return (np.array(sorted(eraser)), np.array(sorted(depcal)),
            np.array(sorted(test)))


def train_probe(Htr, ytr):
    sc = StandardScaler().fit(Htr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Htr), ytr)
    return lambda X: clf.predict_proba(sc.transform(X))[:, 1]


def main():
    out = {}
    for model, fn in CACHES.items():
        z = np.load(PIV / fn, allow_pickle=True)
        Hin, y, Hbs, bs_ds = z["Hin"], z["y"], z["Hbs"], z["bs_ds"]
        rng = np.random.default_rng(42)
        ben = np.where(y == 0)[0]; jb = np.where(y == 1)[0]
        rng.shuffle(ben); rng.shuffle(jb)
        cal_b, te_b = ben[:len(ben) // 2], ben[len(ben) // 2:]
        tr = np.array(sorted(list(cal_b) + list(jb[:len(jb) // 2])))
        te = np.array(sorted(list(te_b) + list(jb[len(jb) // 2:])))
        bs_er, bs_dc, bs_te = strat_3way(bs_ds)

        # LEACE: include ALL 3 sources in eraser fit
        src = ["indist_benign"] + sorted(set(map(str, bs_ds[bs_er])))
        Xfit = np.vstack([Hin[cal_b], Hbs[bs_er]])    # H-1: cal_b only
        zlab = np.array([0] * len(cal_b) +
                        [src.index(str(s)) for s in bs_ds[bs_er]])
        Zoh = np.eye(len(src))[zlab][:, 1:]
        mu, R, rnk = leace(Xfit, Zoh)
        er = lambda X: ((X - mu) @ R.T + mu).astype(np.float32)

        score = train_probe(er(Hin[tr]), y[tr])
        # deploy-tau calibrated on bs_dc (includes ALL 3 sources)
        tau_dep = conformal_tau(score(er(Hbs[bs_dc])))

        # per-source FPR at deploy-tau on the held-out test slice
        bs_ds_str = np.array(list(map(str, bs_ds)))
        per_src = {}
        for s in sorted(set(bs_ds_str)):
            mask = bs_ds_str == s
            test_idx = np.array([i for i in bs_te if mask[i]])
            if len(test_idx) == 0:
                continue
            fpr_at_dep = float((score(er(Hbs[test_idx])) >= tau_dep).mean())
            per_src[s] = round(fpr_at_dep, 4)

        # overall test FPR (all sources together)
        fpr_overall = float((score(er(Hbs[bs_te])) >= tau_dep).mean())
        # in-dist test recall
        s_te = score(er(Hin[te]))
        recall = float((s_te[y[te] == 1] >= tau_dep).mean())

        out[model] = dict(
            leace_rank=rnk,
            tau_dep=round(tau_dep, 4),
            per_source_FPR_at_dep_tau=per_src,
            overall_FPR_at_dep_tau=round(fpr_overall, 4),
            recall_at_dep_tau=round(recall, 4),
        )
        print(f"\n== {model} (ALL sources in dep-cal; rank={rnk}) ==")
        print(f"   tau_dep={tau_dep:.4f}")
        print(f"   per-source held-out test FPR: {per_src}")
        print(f"   overall held-out FPR={out[model]['overall_FPR_at_dep_tau']}  "
              f"recall={out[model]['recall_at_dep_tau']}")

    op = PIV / "loso_orbench_include.json"
    op.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
