#!/usr/bin/env python
"""Deployable operating point: what the paper's *prescribed recipe* actually buys.

Pure re-analysis of the cached pivot hidden states (no GPU, no re-inference).
Reproduces the EXACT pivot split + closed-form LEACE so numbers stay consistent
with Table tab:erase, then quantifies three operating points for the baseline
probe vs the register-erased probe, on each backbone:

  (b) in-distribution-calibrated conformal tau  -> the FAILURE we report
      (huge FPR on diverse benign; reproduces Table tab:erase).
  (c) deployment-calibrated split-conformal tau -> the RECIPE we advocate
      (sec:fix step 2): tau calibrated on a held-out half of *diverse* benign,
      FPR measured on the disjoint other half (must be <= alpha by the
      distribution-free guarantee), and jailbreak recall measured at that tau.

The constructive headline = jailbreak recall at a distribution-free <=5% benign
FPR on diverse traffic, baseline vs register-erased, with paired bootstrap 95%
CIs (1000 resamples). Honest either way: if erasure does not raise recall at
matched guaranteed FPR we say so; the conformal guarantee (B) stands regardless.

Env: pinned in requirements.txt (see `bootstrap-latents` conda env)
  python pivot_deployable_operating_point.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PIV = REPO / "data/emnlp2026/pivot"
CACHES = {"qwen_aligned": "hs_qwen_aligned_L20.npz",
          "llama_aligned": "hs_llama_aligned_L16.npz",
          "mistral_aligned": "hs_mistral_aligned_L16.npz"}
ALPHA = 0.05


def leace(Xfit, Zfit, eps=1e-3):                       # identical to pivot script
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
    return mu.astype(np.float32), R.astype(np.float32), r


def train_probe(Htr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Htr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Htr), ytr)
    return lambda X: clf.predict_proba(sc.transform(X))[:, 1]


def conformal_tau(scores_benign, alpha=ALPHA):
    s = np.sort(scores_benign)
    k = int(np.ceil((len(s) + 1) * (1 - alpha))) - 1
    return float(s[min(k, len(s) - 1)])


def main():
    rng_master = np.random.default_rng(42)
    out = {}
    for model, fn in CACHES.items():
        z = np.load(PIV / fn, allow_pickle=True)
        Hin, y, Hbs, bs_ds = z["Hin"], z["y"], z["Hbs"], z["bs_ds"]
        # ---- EXACT pivot split (so consistent with Table tab:erase) ----
        rng = np.random.default_rng(42)
        ben = np.where(y == 0)[0]; jb = np.where(y == 1)[0]
        rng.shuffle(ben); rng.shuffle(jb)
        cal_b, te_b = ben[:len(ben) // 2], ben[len(ben) // 2:]
        tr = np.array(sorted(list(cal_b) + list(jb[:len(jb) // 2])))
        te = np.array(sorted(list(te_b) + list(jb[len(jb) // 2:])))
        # ---- LEACE eraser, benign-only, source identity (amnesic) ----
        src = ["indist_benign"] + sorted(set(map(str, bs_ds)))
        Xfit = np.vstack([Hin[ben], Hbs])
        zlab = np.array([0] * len(ben) + [src.index(str(s)) for s in bs_ds])
        Zoh = np.eye(len(src))[zlab][:, 1:]
        mu, R, rnk = leace(Xfit, Zoh)
        er = lambda X: ((X - mu) @ R.T + mu).astype(np.float32)

        # disjoint diverse-benign calib/test halves (deployment calibration)
        bidx = np.arange(len(Hbs)); np.random.default_rng(7).shuffle(bidx)
        bcal, btest = bidx[:len(bidx) // 2], bidx[len(bidx) // 2:]
        jte = te[y[te] == 1]                       # held-out jailbreaks
        bte_id = te[y[te] == 0]                    # held-out in-dist benign

        res = {"leace_rank": int(rnk)}
        for name, Xtr, Xin, Xbs in [
                ("baseline", Hin[tr], Hin, Hbs),
                ("register_erased", er(Hin[tr]), er(Hin), er(Hbs))]:
            score = train_probe(Xtr, y[tr])
            s_jte = score(Xin[jte])
            s_bte_id = score(Xin[bte_id])
            s_bcal = score(Xbs[bcal]); s_btest = score(Xbs[btest])
            # (b) in-dist-calibrated tau -> failure on diverse benign
            tau_id = conformal_tau(score(Xin[bte_id]))
            fpr_div_idcal = float((score(Xbs) >= tau_id).mean())
            # (c) deployment-calibrated tau (THE RECIPE)
            tau_dep = conformal_tau(s_bcal)
            fpr_div_depcal = float((s_btest >= tau_dep).mean())   # ~<=0.05
            rec_dep = float((s_jte >= tau_dep).mean())
            rec_idcal = float((s_jte >= tau_id).mean())
            # paired bootstrap CI on recall at deployment tau
            B = 1000; recs = np.empty(B)
            n = len(s_jte)
            for b in range(B):
                idx = rng_master.integers(0, n, n)
                recs[b] = (s_jte[idx] >= tau_dep).mean()
            ci = (float(np.percentile(recs, 2.5)),
                  float(np.percentile(recs, 97.5)))
            res[name] = {
                "tau_indist_calibrated": round(tau_id, 4),
                "FPR_diverse_benign_at_indist_tau": round(fpr_div_idcal, 4),
                "recall_jb_at_indist_tau": round(rec_idcal, 4),
                "tau_deployment_calibrated": round(tau_dep, 4),
                "FPR_diverse_benign_at_deploy_tau": round(fpr_div_depcal, 4),
                "recall_jb_at_deploy_tau": round(rec_dep, 4),
                "recall_jb_at_deploy_tau_CI95": [round(ci[0], 4),
                                                 round(ci[1], 4)]}
        # paired delta (erased - baseline) recall at deployment tau, CI
        sb = train_probe(Hin[tr], y[tr]); se = train_probe(er(Hin[tr]), y[tr])
        Her = er(Hin); Hbs_er = er(Hbs)
        sjb_b = sb(Hin[jte]); sjb_e = se(Her[jte])
        tb = conformal_tau(sb(Hbs[bcal]))            # baseline deployment tau
        td = conformal_tau(se(Hbs_er[bcal]))         # erased  deployment tau
        n = len(sjb_b); B = 1000; d = np.empty(B)
        for b in range(B):
            idx = rng_master.integers(0, n, n)
            d[b] = (sjb_e[idx] >= td).mean() - (sjb_b[idx] >= tb).mean()
        res["delta_recall_erased_minus_baseline_at_deploy_tau"] = {
            "point": round(float((sjb_e >= td).mean()
                                 - (sjb_b >= tb).mean()), 4),
            "CI95": [round(float(np.percentile(d, 2.5)), 4),
                     round(float(np.percentile(d, 97.5)), 4)]}
        out[model] = res
        b, e = res["baseline"], res["register_erased"]
        print(f"\n== {model} (LEACE rank {rnk}) ==")
        print(f"  IN-DIST-calibrated  : diverse-benign FPR "
              f"{b['FPR_diverse_benign_at_indist_tau']} -> "
              f"{e['FPR_diverse_benign_at_indist_tau']} (the failure)")
        print(f"  DEPLOYMENT-calib (recipe): guaranteed diverse-benign FPR "
              f"{b['FPR_diverse_benign_at_deploy_tau']}/"
              f"{e['FPR_diverse_benign_at_deploy_tau']} (<= {ALPHA})")
        print(f"     jailbreak recall @ guaranteed FPR: baseline "
              f"{b['recall_jb_at_deploy_tau']} "
              f"CI{b['recall_jb_at_deploy_tau_CI95']}  vs  erased "
              f"{e['recall_jb_at_deploy_tau']} "
              f"CI{e['recall_jb_at_deploy_tau_CI95']}")
        print("     delta (erased-baseline) "
              f"{res['delta_recall_erased_minus_baseline_at_deploy_tau']}")
    op = PIV / "deployable_operating_point.SUPERSEDED.json"  # CHECK fix C-2: leakage; use heldout_remedy.json
    op.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
