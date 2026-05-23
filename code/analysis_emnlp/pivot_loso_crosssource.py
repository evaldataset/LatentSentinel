#!/usr/bin/env python
"""CHECK.md Critical-2 fix — Leave-One-Source-Out (LOSO) cross-source
generalization of the recipe. Fits the eraser + probe on TWO of the three
benign-stress sources (XSTest / OR-Bench / ToxicChat), evaluates FPR + deploy
recall on the held-out third source the eraser NEVER saw. CPU only.

For each held-out source, for each backbone:
  - eraser-fit set    = in-dist benign + the 2 NON-held-out sources (full)
  - deploy-calib set  = ~half of the 2 NON-held-out sources
  - test eval         = ENTIRELY the held-out source (cross-source)
  - in-dist tau still calibrated on in-dist benign test (same as paper)

Reports: baseline & erased FPR_OOD@indist-tau on the held-out source; deploy tau
calibrated on the 2 non-held sources, FPR + recall@deploy-tau on the held-out
source. Honest: this directly tests "does the recipe generalize across benign
source families?".

Env: pinned in requirements.txt (see `bootstrap-latents` conda env)
  python pivot_loso_crosssource.py
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


def train_probe(Htr, ytr):
    sc = StandardScaler().fit(Htr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Htr), ytr)
    return lambda X: clf.predict_proba(sc.transform(X))[:, 1]


def run_loso(Hin, y, Hbs, bs_ds, held_source):
    bs_str = np.array(list(map(str, bs_ds)))
    held_idx = np.where(bs_str == held_source)[0]
    fit_idx  = np.where(bs_str != held_source)[0]
    rng = np.random.default_rng(20260520)
    rng.shuffle(fit_idx)
    n = len(fit_idx)
    bs_eraser = np.array(sorted(fit_idx[:n // 2]))  # eraser+probe TRAIN benign
    bs_depcal = np.array(sorted(fit_idx[n // 2:]))  # deployment-calibration benign
    bs_test   = held_idx                            # CROSS-SOURCE held-out
    # same in-dist split as pivot_heldout_remedy (seed 42)
    rng2 = np.random.default_rng(42)
    ben = np.where(y == 0)[0]; jb = np.where(y == 1)[0]
    rng2.shuffle(ben); rng2.shuffle(jb)
    cal_b, te_b = ben[:len(ben) // 2], ben[len(ben) // 2:]
    tr = np.array(sorted(list(cal_b) + list(jb[:len(jb) // 2])))
    te = np.array(sorted(list(te_b) + list(jb[len(jb) // 2:])))
    # Z over only the 2 non-held sources (eraser sees only those)
    # CHECK fix H-1: Hin[ben] -> Hin[cal_b] (in-dist test te_b excluded).
    src_in_fit = sorted(set(map(str, bs_ds[bs_eraser])))
    src = ["indist_benign"] + src_in_fit
    Xfit = np.vstack([Hin[cal_b], Hbs[bs_eraser]])
    zlab = np.array([0] * len(cal_b) +
                    [src.index(str(s)) for s in bs_ds[bs_eraser]])
    Zoh = np.eye(len(src))[zlab][:, 1:]
    mu, R, rnk = leace(Xfit, Zoh)
    er = lambda X: ((X - mu) @ R.T + mu).astype(np.float32)

    out = {"leace_rank": rnk,
           "n_held_source_test": int(len(bs_test)),
           "fit_sources": src_in_fit}
    for name, score, Xbs in [
            ("baseline", train_probe(Hin[tr], y[tr]), Hbs),
            ("register_erased", train_probe(er(Hin[tr]), y[tr]), er(Hbs))]:
        s_te = score(Hin[te]); yte = y[te]
        tau_id = conformal_tau(s_te[yte == 0])
        tau_dep = conformal_tau(score(Xbs[bs_depcal]))
        fpr_crosssource_idcal = float((score(Xbs[bs_test]) >= tau_id).mean())
        fpr_crosssource_depcal = float((score(Xbs[bs_test]) >= tau_dep).mean())
        rec = float((s_te[yte == 1] >= tau_dep).mean())
        out[name] = dict(
            FPR_crosssource_at_indist_tau=round(fpr_crosssource_idcal, 4),
            FPR_crosssource_at_deploy_tau=round(fpr_crosssource_depcal, 4),
            recall_at_deploy_tau=round(rec, 4))
    return out


def main():
    out = {}
    for model, fn in CACHES.items():
        z = np.load(PIV / fn, allow_pickle=True)
        Hin, y, Hbs, bs_ds = z["Hin"], z["y"], z["Hbs"], z["bs_ds"]
        sources = sorted(set(map(str, bs_ds)))
        out[model] = {}
        print(f"\n========= {model} (LOSO over {sources}) =========")
        for held in sources:
            r = run_loso(Hin, y, Hbs, bs_ds, held)
            out[model][held] = r
            b, e = r["baseline"], r["register_erased"]
            print(f"  HELD-OUT {held}: rank{r['leace_rank']} | "
                  f"FPR@indist-tau  base {b['FPR_crosssource_at_indist_tau']} "
                  f"-> erased {e['FPR_crosssource_at_indist_tau']} | "
                  f"@deploy-tau  base {b['FPR_crosssource_at_deploy_tau']} / "
                  f"erased {e['FPR_crosssource_at_deploy_tau']} | "
                  f"recall@deploy: base {b['recall_at_deploy_tau']} -> erased "
                  f"{e['recall_at_deploy_tau']}")
    op = PIV / "loso_crosssource.json"
    op.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {op}")
    # summary: cross-source FPR reduction at indist-tau and at deploy-tau
    print("\n=== summary (baseline -> erased on the HELD-OUT source) ===")
    for m, src_res in out.items():
        for s, r in src_res.items():
            d_id = r["baseline"]["FPR_crosssource_at_indist_tau"] - \
                   r["register_erased"]["FPR_crosssource_at_indist_tau"]
            d_dep = r["baseline"]["FPR_crosssource_at_deploy_tau"] - \
                    r["register_erased"]["FPR_crosssource_at_deploy_tau"]
            print(f"  {m:18s} held={s:18s} "
                  f"Δ indist-tau FPR {d_id:+.3f}  Δ deploy-tau FPR {d_dep:+.3f}  "
                  f"erased recall {r['register_erased']['recall_at_deploy_tau']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
