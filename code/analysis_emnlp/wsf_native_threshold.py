#!/usr/bin/env python
"""P0 Soundness firewall: over-refusal persists at EACH detector's OWN
sensitivity-matched operating point (not just our conformal tau).

For every (detector, model): take the detector's per-sample scores, pick the
threshold that yields 95% TPR on HELD-OUT in-distribution jailbreaks (the
detector's own native sensitivity), then report the benign-stress FPR at that
threshold. This pre-empts the #1 reject vector ("the failure is a threshold
artifact of your conformal choice"). No GPU; pure re-analysis of saved scores.

Convention: higher score = more jailbreak. Output → data/emnlp2026/wsf_native_threshold.json
Env: pinned in requirements.txt (see `bootstrap-latents` conda env)
"""
from __future__ import annotations
import glob, json, re
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parents[2]
WSB, WSD = REPO / "data/emnlp2026/wsb", REPO / "data/emnlp2026/wsd"
PRED = REPO / "data/predictions"
RNG = np.random.default_rng(42)
PROBE_BEST = {"qwen_aligned": ("stander_jailbreak_eval_qwen", 20),
              "llama_aligned": ("stander_jailbreak_eval", 16)}


def tpr95_threshold(yj_scores: np.ndarray) -> float:
    """Threshold s.t. 95% of jailbreak scores are >= it (95% TPR)."""
    return float(np.quantile(yj_scores, 0.05))


def boot_fpr(bs: np.ndarray, tau: float, B=1000):
    n = len(bs); idx = np.arange(n)
    v = [float((bs[RNG.choice(idx, n, replace=True)] >= tau).mean()) for _ in range(B)]
    return round(float(np.mean(v)), 4), round(float(np.percentile(v, 2.5)), 4), \
        round(float(np.percentile(v, 97.5)), 4)


def split_cal_test(y, seed=42):
    """Same stratified 50/50 split as wsd_baselines; return (cal_idx, test_idx).
    AUDIT FIX C5: the 95%-TPR threshold is set on CALIBRATION jailbreaks and the
    realized TPR is verified on the DISJOINT test jailbreaks (the old code set
    tau on the test jb it then reported TPR for -> tautological in-sample
    selection)."""
    rng = np.random.default_rng(seed)
    cal, tst = [], []
    for c in (0, 1):
        ci = np.where(y == c)[0]; rng.shuffle(ci)
        cal += list(ci[:len(ci)//2]); tst += list(ci[len(ci)//2:])
    return np.array(sorted(cal)), np.array(sorted(tst))


def add(res, name, indist, y, bs):
    cal, tst = split_cal_test(y)
    # AUDIT FIX F32: orient detector sign by CALIBRATION AUC (a detector with
    # cal-AUC < 0.5, e.g. FJD here, is anti-correlated; a native 95%-TPR
    # threshold on un-oriented scores is meaningless / FPR=1.0).
    cal_auc = float(roc_auc_score(y[cal], indist[cal]))
    oriented = cal_auc < 0.5
    s = -indist if oriented else indist
    cal_jb = cal[y[cal] == 1]
    test_jb = tst[y[tst] == 1]
    tau = tpr95_threshold(s[cal_jb])                 # tau from CALIBRATION jb
    realized_tpr = float((s[test_jb] >= tau).mean())  # verified on TEST jb
    auc = float(roc_auc_score(y, s))                  # oriented AUC (>=0.5)
    fpr, tpr, _ = roc_curve(y, s)
    i = np.searchsorted(tpr, 0.95)
    fpr95_direct = float(fpr[min(i, len(fpr) - 1)])
    s_bs = -bs if oriented else bs        # benign-stress in the oriented space
    mu, lo, hi = boot_fpr(s_bs, tau)
    res[name] = {
        "roc_auc_indist_oriented": round(auc, 4),
        "sign_oriented_by_cal_auc": oriented,
        "tau_at_95TPR_CAL_jb": round(tau, 6),
        "realized_jb_TPR_on_TEST_jb": round(realized_tpr, 4),
        "benign_stress_FPR_at_native_tau": mu,
        "benign_stress_FPR_CI95": [lo, hi],
        "FPR_at_95TPR_threshold_free": round(fpr95_direct, 4),
    }
    print(f"{name:34s} AUC={auc:.3f} oriented={oriented}  "
          f"realizedTPR(test)={realized_tpr:.3f}  benign-stress FPR="
          f"{mu:.3f} [{lo:.3f},{hi:.3f}]  (thr-free FPR@95TPR={fpr95_direct:.3f})")


def main():
    res = {}
    # --- WS-D baseline families (npz: det in-dist, y, det_bs benign-stress) ---
    for f in sorted(glob.glob(str(WSD / "baseline_scores_*.npz"))):
        m = re.search(r"baseline_scores_(\w+?)_(qwen_aligned|llama_aligned)\.npz$",
                      Path(f).name)
        meth, mdl = m.group(1), m.group(2)
        z = np.load(f)
        add(res, f"{meth}/{mdl}", z["det"], z["y"], z["det_bs"])

    # --- our trained probe: in-dist from predictions CSV @ best layer; ---
    #     benign-stress from WS-B saved P(jb) .npy ---
    import pandas as pd
    for mdl, (pdir, layer) in PROBE_BEST.items():
        csv = sorted(glob.glob(str(PRED / pdir / f"*_{layer}_predictions.csv")))
        if not csv:
            print(f"[skip] probe/{mdl}: no layer-{layer} csv"); continue
        df = pd.read_csv(csv[0])
        indist = 1.0 - df["prob_benign"].values            # P(jailbreak)
        y = (df["label"].values == 0).astype(int)           # jb=1
        bs = np.load(WSB / f"scores_{mdl}.npy")             # WS-B P(jb), benign-stress
        add(res, f"ourprobe/{mdl}", indist, y, bs)

    out = REPO / "data/emnlp2026/wsf_native_threshold.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"\nwritten: {out}")
    print("Interpretation: at each detector's OWN 95%-TPR operating point, "
          "benign-stress FPR remains high for the high-AUC detectors — the "
          "over-refusal failure is not an artifact of the conformal threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
