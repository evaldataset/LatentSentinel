#!/usr/bin/env python
"""A3 — bootstrap 95% CIs for the main-table cells (tab:erase erased FPR_OOD,
tab:deploy erased recall) on the held-out partition. CPU only, 1000 resamples.
Re-uses the EXACT pivot_heldout_remedy split + LEACE protocol.
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
    return (mu.astype(np.float32),
            (sq @ (np.eye(d) - Q @ Q.T) @ isq).astype(np.float32))


def conformal_tau(s, alpha=ALPHA):
    s = np.sort(s); k = int(np.ceil((len(s) + 1) * (1 - alpha))) - 1
    return float(s[min(k, len(s) - 1)])


def strat_3way(bs_ds, seed=20260520):
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
    BOOT = 1000
    rng = np.random.default_rng(20260520)
    for model, fn in CACHES.items():
        z = np.load(PIV / fn, allow_pickle=True)
        Hin, y, Hbs, bs_ds = z["Hin"], z["y"], z["Hbs"], z["bs_ds"]
        rng2 = np.random.default_rng(42)
        ben = np.where(y == 0)[0]; jb = np.where(y == 1)[0]
        rng2.shuffle(ben); rng2.shuffle(jb)
        cal_b, te_b = ben[:len(ben) // 2], ben[len(ben) // 2:]
        tr = np.array(sorted(list(cal_b) + list(jb[:len(jb) // 2])))
        te = np.array(sorted(list(te_b) + list(jb[len(jb) // 2:])))
        bs_er, bs_dc, bs_te = strat_3way(bs_ds)
        src = ["indist_benign"] + sorted(set(map(str, bs_ds[bs_er])))
        # CHECK fix H-1: LEACE fit uses Hin[cal_b] only (excludes in-dist test).
        Xfit = np.vstack([Hin[cal_b], Hbs[bs_er]])
        zlab = np.array([0] * len(cal_b) +
                        [src.index(str(s)) for s in bs_ds[bs_er]])
        Zoh = np.eye(len(src))[zlab][:, 1:]
        mu, R = leace(Xfit, Zoh)
        er = lambda X: ((X - mu) @ R.T + mu).astype(np.float32)
        Htr2 = np.vstack([Hin[tr], Hbs[bs_er]])
        ytr2 = np.concatenate([y[tr], np.zeros(len(bs_er), int)])

        results = {}
        for nm, score, Xin, Xbs in [
                ("baseline", train_probe(Hin[tr], y[tr]), Hin, Hbs),
                ("erased",   train_probe(er(Hin[tr]), y[tr]), er(Hin), er(Hbs)),
                ("regbal",   train_probe(Htr2, ytr2), Hin, Hbs)]:
            yte = y[te]; s_te = score(Xin[te])
            tau_id = conformal_tau(s_te[yte == 0])
            tau_dep = conformal_tau(score(Xbs[bs_dc]))
            s_bs_test = score(Xbs[bs_te])
            s_jb_te = s_te[yte == 1]
            # bootstrap CIs by resampling the test sets
            fpr_id_b = []; fpr_dep_b = []; rec_dep_b = []
            n_bs = len(s_bs_test); n_jb = len(s_jb_te)
            for _ in range(BOOT):
                ii = rng.integers(0, n_bs, n_bs)
                fpr_id_b.append(float((s_bs_test[ii] >= tau_id).mean()))
                fpr_dep_b.append(float((s_bs_test[ii] >= tau_dep).mean()))
                jj = rng.integers(0, n_jb, n_jb)
                rec_dep_b.append(float((s_jb_te[jj] >= tau_dep).mean()))
            results[nm] = {
                "indistcal_FPR_OOD":   {"pt": round(float((s_bs_test >= tau_id).mean()), 4),
                                        "CI95": [round(float(np.percentile(fpr_id_b, 2.5)), 4),
                                                  round(float(np.percentile(fpr_id_b, 97.5)), 4)]},
                "deploy_FPR_OOD":      {"pt": round(float((s_bs_test >= tau_dep).mean()), 4),
                                        "CI95": [round(float(np.percentile(fpr_dep_b, 2.5)), 4),
                                                  round(float(np.percentile(fpr_dep_b, 97.5)), 4)]},
                "deploy_recall_jb":    {"pt": round(float((s_jb_te >= tau_dep).mean()), 4),
                                        "CI95": [round(float(np.percentile(rec_dep_b, 2.5)), 4),
                                                  round(float(np.percentile(rec_dep_b, 97.5)), 4)]}}
        out[model] = results
        b = results["baseline"]; e = results["erased"]; rb = results["regbal"]
        print(f"\n{model}:")
        print(f"  tab:erase  FPR_OOD@indist-tau  base {b['indistcal_FPR_OOD']['pt']}{b['indistcal_FPR_OOD']['CI95']} "
              f"erased {e['indistcal_FPR_OOD']['pt']}{e['indistcal_FPR_OOD']['CI95']} "
              f"regbal {rb['indistcal_FPR_OOD']['pt']}{rb['indistcal_FPR_OOD']['CI95']}")
        print(f"  tab:deploy recall@deploy-tau  base {b['deploy_recall_jb']['pt']}{b['deploy_recall_jb']['CI95']} "
              f"erased {e['deploy_recall_jb']['pt']}{e['deploy_recall_jb']['CI95']}")
    op = PIV / "bootstrap_main_tables.json"
    op.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
