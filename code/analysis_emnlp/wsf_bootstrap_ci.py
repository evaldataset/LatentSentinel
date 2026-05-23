#!/usr/bin/env python
"""WS-F: bootstrap 95% CIs on the headline numbers (no GPU, pure re-analysis).

Resamples the saved per-prompt score arrays (1000x) to put confidence intervals on:
 - WS-D head-to-head: in-dist ROC-AUC (resample test set) and benign-stress FPR @ the
   FIXED stored conformal tau (resample stress set), for every method x model in
   data/emnlp2026/wsd/baseline_scores_*.npz + scores_*.npz (HiddenDetect-style) +
   our probe's WS-B over-refusal (data/emnlp2026/wsb/scores_*.npy).
 - WS-C: universal-suffix evasion rate (resample held-out jailbreaks).
All thresholds are the ones already calibrated on held-out in-dist benign (not refit).

Env: pinned in requirements.txt (see `bootstrap-latents` conda env)
  python wsf_bootstrap_ci.py
"""
from __future__ import annotations
import glob, json, os, re
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
WSB, WSC, WSD = (REPO / "data/emnlp2026/wsb", REPO / "data/emnlp2026/wsc",
                 REPO / "data/emnlp2026/wsd")
RNG = np.random.default_rng(42)
B = 1000


def ci(vals):
    a = np.asarray(vals, float)
    return round(float(a.mean()), 4), round(float(np.percentile(a, 2.5)), 4), \
        round(float(np.percentile(a, 97.5)), 4)


def boot_auc(y, s):
    out = []
    n = len(y)
    idx = np.arange(n)
    for _ in range(B):
        b = RNG.choice(idx, n, replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        out.append(roc_auc_score(y[b], s[b]))
    return ci(out)


def boot_rate(mask_scores, tau):
    out = []
    n = len(mask_scores)
    idx = np.arange(n)
    for _ in range(B):
        b = RNG.choice(idx, n, replace=True)
        out.append(float((mask_scores[b] >= tau).mean()))
    return ci(out)


def main():
    res = {}

    # ---- WS-D baselines (fjd/gradsafe/jbshield) + HiddenDetect-style ----
    for f in sorted(glob.glob(str(WSD / "baseline_scores_*.npz"))):
        m = re.search(r"baseline_scores_(\w+?)_(qwen_aligned|llama_aligned)\.npz$",
                      os.path.basename(f))
        meth, mdl = m.group(1), m.group(2)
        z = np.load(f)
        det, y, det_bs = z["det"], z["y"], z["det_bs"]
        tst, tau = z["tst"], float(z["tau"])
        res[f"{meth}/{mdl}"] = {
            "auc_indist": boot_auc(y[tst], det[tst]),
            "benign_stress_FPR@tau": boot_rate(det_bs, tau),
        }
    # NOTE: HiddenDetect-style (scores_*.npz) is SUPERSEDED by the faithful
    # hiddendetect_exact (its baseline_scores_*.npz is globbed above) — dropped.

    # Spearman rho: in-dist AUC vs OOD benign FPR across all (detector,model) cells
    from scipy.stats import spearmanr
    pts = []
    bj = json.loads((WSD / "baselines.json").read_text())
    for meth, mm in bj.items():
        for mdl, r in mm.items():
            pts.append((r["roc_auc_indist"], r["benign_stress_FPR_overall"],
                        f"{meth}/{mdl}"))
    aucs = [p[0] for p in pts]; fprs = [p[1] for p in pts]
    rho, pval = spearmanr(aucs, fprs)
    res["_tradeoff_spearman"] = {
        "rho": round(float(rho), 4), "p_value": float(f"{pval:.2e}"),
        "n_cells": len(pts),
        "points": sorted([(round(a, 3), round(f, 3), n) for a, f, n in pts])}

    # ---- our probe WS-B over-refusal (the headline) ----
    bfpr = json.loads((WSB / "benign_stress_fpr.json").read_text())
    for npy in sorted(glob.glob(str(WSB / "scores_*_aligned.npy"))):
        mdl = re.search(r"scores_(\w+_aligned)\.npy$", npy).group(1)
        s = np.load(npy)
        key = "qwen_aligned" if "qwen" in mdl else "llama_aligned"
        tau = float(bfpr[key]["thresholds"]["conformal_alpha_0.05"])
        res[f"OURPROBE_overrefusal/{mdl}"] = {
            "benign_stress_FPR@conf_tau": boot_rate(s, tau)}

    # ---- WS-C evasion rate ----
    for f in sorted(glob.glob(str(WSC / "scores_*_aligned.npz"))):
        mdl = re.search(r"scores_(\w+_aligned)\.npz$", f).group(1)
        z = np.load(f)
        clean, adv = z["clean"], z["adv"]
        ae = json.loads((WSC / "adaptive_evasion.json").read_text())
        key = "qwen_aligned" if "qwen" in mdl else "llama_aligned"
        tau = float(ae[key]["tau_conformal_a0.05"])
        caught = clean >= tau
        ev = []
        idx = np.arange(len(clean))
        for _ in range(B):
            b = RNG.choice(idx, len(clean), replace=True)
            c = caught[b]
            ev.append(float(((adv[b] < tau) & c).sum() / max(c.sum(), 1)))
        res[f"WSC_evasion/{mdl}"] = {"evasion_rate": ci(ev)}

    out = WSD.parent / "wsf_bootstrap_ci.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"Bootstrap 95% CIs (B={B}) — mean [lo, hi]\n" + "=" * 64)
    for k, v in res.items():
        if k.startswith("_"):
            print(f"{k}: {json.dumps(v)[:300]}")
            continue
        print(f"{k}")
        for mname, (mu, lo, hi) in v.items():
            print(f"   {mname:28s} {mu:.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
