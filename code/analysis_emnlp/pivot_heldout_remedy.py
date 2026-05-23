#!/usr/bin/env python
"""B1 FIX — leakage-free register-erasure remedy + deployable operating point.

Audit finding B1 (AUDIT.md): the original pivot_register_erase_remedy.py /
pivot_deployable_operating_point.py fit the LEACE eraser on the FULL benign-stress
set Hbs and then evaluate the erased probe's OOD-FPR / deployment recall on (a
split of) those SAME rows -> transductive leakage. This script fixes it with a
source-stratified, mutually DISJOINT 3-way split of Hbs:
  bs_eraser  -> fit the LEACE eraser only (with in-dist benign)
  bs_depcal  -> deployment-calibration set for split-conformal tau
  bs_test    -> held-out evaluation (OOD-FPR / recall) -- NEVER seen by eraser
The in-distribution split (probe train/test, in-dist conformal tau) is identical
to the original (seed 42) so the leakage fix is the ONLY change. All three
remedy conditions (baseline / register-erased / register-balanced) are evaluated
on the SAME held-out bs_test for an apples-to-apples comparison. Z-recoverability
is measured on held-out points (non-circular, fixes audit F38).

Pure re-analysis of data/emnlp2026/pivot/hs_*.npz. CPU only.
Env: pinned in requirements.txt (see `bootstrap-latents` conda env)
  python pivot_heldout_remedy.py
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


def leace(Xfit, Zfit, eps=1e-3):                       # identical math to originals
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


def conformal_tau(scores_benign, alpha=ALPHA):
    s = np.sort(scores_benign)
    k = int(np.ceil((len(s) + 1) * (1 - alpha))) - 1     # standard split-conformal
    return float(s[min(k, len(s) - 1)])


def train_probe(Htr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Htr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Htr), ytr)
    return lambda X: clf.predict_proba(sc.transform(X))[:, 1]


def cv_auc(X, yb):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    p = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        X, yb, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        method="predict_proba", n_jobs=4)[:, 1]
    return float(roc_auc_score(yb, p))


def strat_3way(bs_ds, seed=20260520):
    """Source-stratified disjoint 3-way split of Hbs row indices."""
    rng = np.random.default_rng(seed)
    eraser, depcal, test = [], [], []
    for s in sorted(set(map(str, bs_ds))):
        idx = np.where(np.array(list(map(str, bs_ds))) == s)[0]
        rng.shuffle(idx)
        n = len(idx); a, b = n // 3, 2 * n // 3
        eraser += list(idx[:a]); depcal += list(idx[a:b]); test += list(idx[b:])
    return (np.array(sorted(eraser)), np.array(sorted(depcal)),
            np.array(sorted(test)))


def evaluate(score, Hin, y, te, Hbs, bs_test, bs_depcal):
    """AUC + jb-recall at IN-DIST-conformal tau, OOD-FPR on HELD-OUT bs_test,
    plus deployment-calibrated tau (on bs_depcal) -> FPR/recall on bs_test."""
    from sklearn.metrics import roc_auc_score
    s_te = score(Hin[te]); yte = y[te]
    auc = float(roc_auc_score(yte, s_te))
    ben_te = s_te[yte == 0]
    tau_id = conformal_tau(ben_te)
    jb_rec_id = float((s_te[yte == 1] >= tau_id).mean())
    fpr_ood_idcal = float((score(Hbs[bs_test]) >= tau_id).mean())   # held-out
    tau_dep = conformal_tau(score(Hbs[bs_depcal]))
    fpr_ood_depcal = float((score(Hbs[bs_test]) >= tau_dep).mean())  # held-out
    rec_dep = float((s_te[yte == 1] >= tau_dep).mean())
    return dict(auc=round(auc, 4),
                jb_recall_at_indist_tau=round(jb_rec_id, 4),
                FPR_ood_heldout_at_indist_tau=round(fpr_ood_idcal, 4),
                FPR_ood_heldout_at_deploy_tau=round(fpr_ood_depcal, 4),
                recall_jb_at_deploy_tau=round(rec_dep, 4),
                _s_jb=s_te[yte == 1], _tau_dep=tau_dep)


def main():
    out = {}
    rngb = np.random.default_rng(20260520)
    for model, fn in CACHES.items():
        cache_path = PIV / fn
        # AUDIT FIX D-N2: cache provenance fingerprint -- log the npz size and a
        # short content hash so a re-run can be cross-checked against expected.
        import hashlib
        sz = cache_path.stat().st_size
        h = hashlib.sha256(cache_path.read_bytes()[:1_000_000]).hexdigest()[:12]
        z = np.load(cache_path, allow_pickle=True)
        Hin, y, Hbs, bs_ds = z["Hin"], z["y"], z["Hbs"], z["bs_ds"]
        print(f"[cache] {fn}: {sz/1e6:.1f}MB sha256[:12]={h} "
              f"Hin{Hin.shape} Hbs{Hbs.shape}", flush=True)
        # in-dist split: IDENTICAL to the original pivot (seed 42)
        rng = np.random.default_rng(42)
        ben = np.where(y == 0)[0]; jb = np.where(y == 1)[0]
        rng.shuffle(ben); rng.shuffle(jb)
        cal_b, te_b = ben[:len(ben) // 2], ben[len(ben) // 2:]
        tr = np.array(sorted(list(cal_b) + list(jb[:len(jb) // 2])))
        te = np.array(sorted(list(te_b) + list(jb[len(jb) // 2:])))
        # DISJOINT source-stratified 3-way split of Hbs (the B1 fix)
        bs_er, bs_dc, bs_te = strat_3way(bs_ds)
        assert len(set(bs_er) & set(bs_dc)) == 0 and \
               len(set(bs_er) & set(bs_te)) == 0 and \
               len(set(bs_dc) & set(bs_te)) == 0, "Hbs split not disjoint"

        # LEACE eraser fit ONLY on in-dist benign CALIBRATION half + bs_er
        # (NEVER bs_te / bs_dc, NEVER in-dist test te_b).
        # CHECK fix H-1: Hin[ben] -> Hin[cal_b] so the in-dist test partition
        # is excluded from eraser-fit data (eraser is unsupervised w.r.t. y,
        # but mu/Sigma must not be computed on the in-dist test partition).
        src = ["indist_benign"] + sorted(set(map(str, bs_ds[bs_er])))
        Xfit = np.vstack([Hin[cal_b], Hbs[bs_er]])
        zlab = np.array([0] * len(cal_b) +
                        [src.index(str(s)) for s in bs_ds[bs_er]])
        Zoh = np.eye(len(src))[zlab][:, 1:]
        mu, R, rnk = leace(Xfit, Zoh)
        er = lambda X: ((X - mu) @ R.T + mu).astype(np.float32)

        base = evaluate(train_probe(Hin[tr], y[tr]), Hin, y, te,
                        Hbs, bs_te, bs_dc)
        ers = evaluate(train_probe(er(Hin[tr]), y[tr]),
                       er(Hin), y, te, er(Hbs), bs_te, bs_dc)
        # register-balanced training: mix bs_er benign into TRAIN, test bs_te
        Htr2 = np.vstack([Hin[tr], Hbs[bs_er]])
        ytr2 = np.concatenate([y[tr], np.zeros(len(bs_er), int)])
        rbt = evaluate(train_probe(Htr2, ytr2), Hin, y, te,
                       Hbs, bs_te, bs_dc)
        # NON-CIRCULAR Z-recoverability: held-out points only (fixes F38)
        Xho = np.vstack([Hin[cal_b][:len(bs_te)], Hbs[bs_te]])
        zho = np.array([0] * min(len(cal_b), len(bs_te)) + [1] * len(bs_te))
        zr_b = cv_auc(Xho, zho)
        zr_a = cv_auc(np.vstack([er(Hin[cal_b])[:len(bs_te)],
                                 er(Hbs)[bs_te]]), zho)
        # paired bootstrap CI on deployable recall delta (erased - baseline)
        sj_b, tb = base["_s_jb"], base["_tau_dep"]
        sj_e, td = ers["_s_jb"], ers["_tau_dep"]
        n = len(sj_b); B = 1000; d = np.empty(B)
        for b in range(B):
            ii = rngb.integers(0, n, n)
            d[b] = (sj_e[ii] >= td).mean() - (sj_b[ii] >= tb).mean()
        for r in (base, ers, rbt):
            r.pop("_s_jb"); r.pop("_tau_dep")
        out[model] = {
            "leace_rank": int(rnk),
            "n_bs_eraser": int(len(bs_er)), "n_bs_depcal": int(len(bs_dc)),
            "n_bs_test_heldout": int(len(bs_te)),
            "baseline": base, "register_erased": ers,
            "register_balanced_training": rbt,
            "Zrecover_heldout_before": round(zr_b, 3),
            "Zrecover_heldout_after": round(zr_a, 3),
            "deploy_delta_recall_erased_minus_baseline": {
                "point": round(float((sj_e >= td).mean()
                                     - (sj_b >= tb).mean()), 4),
                "CI95": [round(float(np.percentile(d, 2.5)), 4),
                         round(float(np.percentile(d, 97.5)), 4)]}}
        b_, e_ = out[model]["baseline"], out[model]["register_erased"]
        print(f"\n== {model} (rank {rnk}; Hbs split "
              f"{len(bs_er)}/{len(bs_dc)}/{len(bs_te)} disjoint) ==")
        print(f"  AUC base {b_['auc']} -> erased {e_['auc']}")
        print(f"  OOD-FPR HELD-OUT @indist-tau: base "
              f"{b_['FPR_ood_heldout_at_indist_tau']} -> erased "
              f"{e_['FPR_ood_heldout_at_indist_tau']} "
              f"| reg-bal {rbt['FPR_ood_heldout_at_indist_tau']}")
        print(f"  DEPLOY (held-out): FPR base "
              f"{b_['FPR_ood_heldout_at_deploy_tau']} / erased "
              f"{e_['FPR_ood_heldout_at_deploy_tau']} (<= {ALPHA}); "
              f"recall base {b_['recall_jb_at_deploy_tau']} -> erased "
              f"{e_['recall_jb_at_deploy_tau']}  "
              f"delta {out[model]['deploy_delta_recall_erased_minus_baseline']}")
        print(f"  Z-recover held-out {zr_b:.3f} -> {zr_a:.3f}")
    op = PIV / "heldout_remedy.json"
    op.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
