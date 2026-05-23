#!/usr/bin/env python
"""AUDIT FIX C4 + M1 + M2 — uniform-protocol accuracy/over-refusal trade-off.

C4: every detector AND the proposed probe scored under ONE identical, label-free
    protocol: sign oriented by calibration AUC; in-dist ROC-AUC on the test half;
    OOD benign-stress FPR at a split-conformal tau (alpha=0.05) calibrated on the
    SAME calibration-benign discipline for all. Removes the calibration-budget
    asymmetry the audit flagged (FJD alone previously got a labeled temperature
    grid-search; the proposed probe got only a label-free tau).
M1: emits the 12-cell Spearman rho, per-model rho, and leave-one-detector-out
    rho range as a committed artifact (previously hand-computed; the only stored
    artifact disagreed at 0.8171/10 cells).
M2: writes per-cell (auc, ood_fpr, ci) so make_tradeoff_figure.py can read the
    probe points from data instead of hardcoding them.

Pure re-analysis of cached scores. CPU only.
  python wsf_tradeoff_uniform.py
"""
from __future__ import annotations
import glob, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
WSB, WSD = REPO / "data/emnlp2026/wsb", REPO / "data/emnlp2026/wsd"
PRED = REPO / "data/predictions"
RNG = np.random.default_rng(42)
ALPHA = 0.05
PROBE = {"qwen_aligned": ("stander_jailbreak_eval_qwen", 20),
         "llama_aligned": ("stander_jailbreak_eval", 16)}


def conformal_tau(s, alpha=ALPHA):
    s = np.sort(s)
    k = int(np.ceil((len(s) + 1) * (1 - alpha))) - 1
    return float(s[min(k, len(s) - 1)])


def split_cal_test(y, seed=42):
    rng = np.random.default_rng(seed); cal, tst = [], []
    for c in (0, 1):
        ci = np.where(y == c)[0]; rng.shuffle(ci)
        cal += list(ci[:len(ci) // 2]); tst += list(ci[len(ci) // 2:])
    return np.array(sorted(cal)), np.array(sorted(tst))


def cell(det, y, det_bs):
    """Uniform label-free protocol -> (auc_indist, ood_fpr, ci)."""
    cal, tst = split_cal_test(y)
    oriented = roc_auc_score(y[cal], det[cal]) < 0.5
    s = -det if oriented else det
    s_bs = -det_bs if oriented else det_bs
    auc = float(roc_auc_score(y[tst], s[tst]))
    tau = conformal_tau(s[cal][y[cal] == 0])          # label-free: benign quantile
    fpr = float((s_bs >= tau).mean())
    n = len(s_bs)
    bb = [float((s_bs[RNG.choice(n, n, replace=True)] >= tau).mean())
          for _ in range(1000)]
    return (round(auc, 4), round(fpr, 4),
            [round(float(np.percentile(bb, 2.5)), 4),
             round(float(np.percentile(bb, 97.5)), 4)], bool(oriented))


def main():
    cells = {}
    for f in sorted(glob.glob(str(WSD / "baseline_scores_*.npz"))):
        m = re.search(r"baseline_scores_(\w+?)_(qwen_aligned|llama_aligned)\.npz$",
                      Path(f).name)
        meth, mdl = m.group(1), m.group(2)
        z = np.load(f)
        a, fp, ci, ori = cell(z["det"], z["y"], z["det_bs"])
        cells[f"{meth}|{mdl}"] = dict(detector=meth, model=mdl, auc=a,
                                      ood_fpr=fp, ci=ci, sign_oriented=ori)
    for mdl, (pdir, layer) in PROBE.items():
        csv = sorted(glob.glob(str(PRED / pdir / f"*_{layer}_predictions.csv")))[0]
        df = pd.read_csv(csv)
        det = 1.0 - df["prob_benign"].values
        y = (df["label"].values == 0).astype(int)
        det_bs = np.load(WSB / f"scores_{mdl}.npy")
        a, fp, ci, ori = cell(det, y, det_bs)
        cells[f"ourprobe|{mdl}"] = dict(detector="ourprobe", model=mdl, auc=a,
                                        ood_fpr=fp, ci=ci, sign_oriented=ori)

    aucs = np.array([c["auc"] for c in cells.values()])
    fprs = np.array([c["ood_fpr"] for c in cells.values()])
    rho, p = spearmanr(aucs, fprs)
    per_model = {}
    for mdl in ("qwen_aligned", "llama_aligned"):
        ci_ = [c for c in cells.values() if c["model"] == mdl]
        r, _ = spearmanr([c["auc"] for c in ci_], [c["ood_fpr"] for c in ci_])
        per_model[mdl] = round(float(r), 4)
    dets = sorted({c["detector"] for c in cells.values()})
    loo = []
    for d in dets:
        sub = [c for c in cells.values() if c["detector"] != d]
        r, _ = spearmanr([c["auc"] for c in sub], [c["ood_fpr"] for c in sub])
        loo.append(round(float(r), 4))
    out = {
        "protocol": "uniform label-free split-conformal a=0.05, sign oriented "
                    "by calibration AUC, identical for every detector incl. the "
                    "proposed probe (AUDIT FIX C4)",
        "n_cells": len(cells),
        "spearman_rho_all_cells": round(float(rho), 4),
        "spearman_p": float(p),
        "per_model_rho": per_model,
        "leave_one_detector_out_rho": {"values": sorted(loo),
                                       "range": [min(loo), max(loo)]},
        "cells": cells,
    }
    op = REPO / "data/emnlp2026/wsf_tradeoff_uniform.json"
    op.write_text(json.dumps(out, indent=1))
    print(f"n_cells={len(cells)}  rho={rho:.4f} (p={p:.2e})  "
          f"per_model={per_model}  LOO range=[{min(loo)},{max(loo)}]")
    for k, c in sorted(cells.items()):
        print(f"  {k:28s} AUC={c['auc']:.3f}  OOD-FPR={c['ood_fpr']:.3f} "
              f"CI{c['ci']}  oriented={c['sign_oriented']}")
    print(f"written: {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
