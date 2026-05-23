#!/usr/bin/env python
"""Extend wsf_tradeoff_uniform to 3-backbone (Mistral included).

Computes Spearman rho over the now-15 detector-x-backbone cells (5 detectors
{Arditi, HiddenDetect, GradSafe, JBShield, FJD} + Our probe = 6 detectors,
2 backbones with all 6 + Mistral with 5 (no BERT Mistral fine-tune) =
15 cells). Reports per-cell AUC + OOD-FPR + bootstrap 95% CI under the
uniform protocol used by wsf_tradeoff_uniform.py.

Adds Mistral cells from:
  - `baseline_scores_{fjd,gradsafe,jbshield}_mistral_aligned.npz` (wsd_baselines)
  - `scores_mistral_aligned.npz` (HiddenDetect different naming convention)
  - audit-fix Mistral probe applied to cached hs (our probe Mistral)
  - `arditi_refusal_summary.json` (Mistral row from G7 baseline)

Output: data/emnlp2026/wsf_tradeoff_3backbone.json + summary printout.
CPU only.
"""
from __future__ import annotations
import glob, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
WSB = REPO / "data/emnlp2026/wsb"
WSD = REPO / "data/emnlp2026/wsd"
PIV = REPO / "data/emnlp2026/pivot"
FIXED = REPO / "data/trained_probes_fixed"
PRED = REPO / "data/predictions"
RNG = np.random.default_rng(42)
ALPHA = 0.05
BACKBONES = ["qwen_aligned", "llama_aligned", "mistral_aligned"]


def conformal_tau(s, alpha=ALPHA):
    s = np.sort(s); k = int(np.ceil((len(s) + 1) * (1 - alpha))) - 1
    return float(s[min(k, len(s) - 1)])


def split_cal_test(y, seed=42):
    rng = np.random.default_rng(seed); cal, tst = [], []
    for c in (0, 1):
        ci = np.where(y == c)[0]; rng.shuffle(ci)
        cal += list(ci[:len(ci) // 2]); tst += list(ci[len(ci) // 2:])
    return np.array(sorted(cal)), np.array(sorted(tst))


def cell(det, y, det_bs):
    cal, tst = split_cal_test(y)
    auc_cal = roc_auc_score(y[cal], det[cal])
    oriented = auc_cal < 0.5
    if oriented:
        det = -det; det_bs = -det_bs
        auc_cal = 1.0 - auc_cal
    auc = float(roc_auc_score(y[tst], det[tst]))
    ben_cal = det[cal][y[cal] == 0]
    tau = conformal_tau(ben_cal, ALPHA)
    fpr = float((det_bs >= tau).mean())
    # bootstrap CI on bs FPR
    n = len(det_bs)
    bb = [float((det_bs[RNG.choice(n, n, replace=True)] >= tau).mean())
          for _ in range(1000)]
    return (round(auc, 4), round(fpr, 4),
            [round(float(np.percentile(bb, 2.5)), 4),
             round(float(np.percentile(bb, 97.5)), 4)], bool(oriented))


def load_arditi_from_cache(model):
    """Reproduce Arditi cell from cached hs and refusal direction."""
    fn = f"hs_{model}_L{20 if model=='qwen_aligned' else 16}.npz"
    z = np.load(PIV / fn, allow_pickle=True)
    Hin, y, Hbs = z["Hin"], z["y"], z["Hbs"]
    cal, tst = split_cal_test(y)
    mu_harmful  = Hin[cal][y[cal] == 1].mean(0)
    mu_harmless = Hin[cal][y[cal] == 0].mean(0)
    d = mu_harmful - mu_harmless
    d = (d / (np.linalg.norm(d) + 1e-12)).astype(np.float32)
    det = Hin @ d; det_bs = Hbs @ d
    return cell(det, y.astype(int), det_bs)


def load_ourprobe_mistral():
    """Our audit-fix probe Mistral on the cached hs."""
    z = np.load(PIV / "hs_mistral_aligned_L16.npz", allow_pickle=True)
    Hin, y, Hbs = z["Hin"], z["y"], z["Hbs"]
    pr = np.load(FIXED / "mistral_aligned/probe_L16_seed42.npz", allow_pickle=True)
    w = pr["w"].astype(np.float32); b = float(pr["b"])
    det = 1.0 / (1.0 + np.exp(-(Hin @ w + b)))
    det_bs = 1.0 / (1.0 + np.exp(-(Hbs @ w + b)))
    return cell(det, y.astype(int), det_bs)


def main():
    cells = {}

    # Baselines from baseline_scores_*.npz (FJD, GradSafe, JBShield, BERT)
    for f in sorted(glob.glob(str(WSD / "baseline_scores_*.npz"))):
        m = re.search(r"baseline_scores_(.+?)_(qwen_aligned|llama_aligned|mistral_aligned)\.npz$",
                      Path(f).name)
        if not m:
            continue
        meth, mdl = m.group(1), m.group(2)
        if meth == "hiddendetect_exact":
            meth_norm = "hiddendetect"
        else:
            meth_norm = meth
        z = np.load(f)
        a, fp, ci, ori = cell(z["det"], z["y"], z["det_bs"])
        cells[f"{meth_norm}|{mdl}"] = dict(detector=meth_norm, model=mdl,
                                            auc=a, ood_fpr=fp, ci=ci,
                                            sign_oriented=ori)

    # HiddenDetect on Mistral (different file naming)
    fp_mhd = WSD / "scores_mistral_aligned.npz"
    if fp_mhd.exists() and "hiddendetect|mistral_aligned" not in cells:
        z = np.load(fp_mhd)
        # check columns
        keys = list(z.keys())
        # File should have det / y / det_bs (per wsd_hiddendetect_baseline.py savez)
        if "indist" in keys and "y" in keys and "bstress" in keys:
            a, fp, ci, ori = cell(z["indist"], z["y"], z["bstress"])
            cells[f"hiddendetect|mistral_aligned"] = dict(
                detector="hiddendetect", model="mistral_aligned",
                auc=a, ood_fpr=fp, ci=ci, sign_oriented=ori)

    # Arditi (computed from cached hs each backbone)
    for mdl in BACKBONES:
        a, fp, ci, ori = load_arditi_from_cache(mdl)
        cells[f"arditi|{mdl}"] = dict(detector="arditi", model=mdl,
                                       auc=a, ood_fpr=fp, ci=ci,
                                       sign_oriented=ori)

    # Our probe on Mistral (audit-fix retrained, no legacy probe exists)
    a, fp, ci, ori = load_ourprobe_mistral()
    cells["ourprobe|mistral_aligned"] = dict(detector="ourprobe",
                                              model="mistral_aligned",
                                              auc=a, ood_fpr=fp, ci=ci,
                                              sign_oriented=ori)

    # Inherit Qwen/Llama "ourprobe" from existing wsf JSON (no change for them)
    prev_json = REPO / "data/emnlp2026/wsf_tradeoff_uniform.json"
    if prev_json.exists():
        prev = json.loads(prev_json.read_text())
        for k, c in prev.get("cells", {}).items():
            if k.startswith("ourprobe|") and k.endswith(("qwen_aligned", "llama_aligned")):
                cells[k] = c

    aucs = np.array([c["auc"] for c in cells.values()])
    fprs = np.array([c["ood_fpr"] for c in cells.values()])
    rho, p = spearmanr(aucs, fprs)
    per_model = {}
    for mdl in BACKBONES:
        ci_ = [c for c in cells.values() if c["model"] == mdl]
        if len(ci_) >= 3:
            r, _ = spearmanr([c["auc"] for c in ci_], [c["ood_fpr"] for c in ci_])
            per_model[mdl] = round(float(r), 4)

    dets = sorted({c["detector"] for c in cells.values()})
    loo = []
    for d in dets:
        sub = [c for c in cells.values() if c["detector"] != d]
        r, _ = spearmanr([c["auc"] for c in sub], [c["ood_fpr"] for c in sub])
        loo.append(round(float(r), 4))

    out = {
        "protocol": "uniform label-free split-conformal a=0.05; 3-backbone extension",
        "n_cells": len(cells),
        "spearman_rho_all_cells": round(float(rho), 4),
        "spearman_p": float(p),
        "per_model_rho": per_model,
        "leave_one_detector_out_rho": {
            "values": sorted(loo),
            "range": [min(loo), max(loo)]},
        "cells": cells,
    }
    op = REPO / "data/emnlp2026/wsf_tradeoff_3backbone.json"
    op.write_text(json.dumps(out, indent=1))
    print(f"\nn_cells={len(cells)}  rho_all={rho:.4f} (p={p:.2e})")
    print(f"per_model_rho={per_model}")
    print(f"LOO rho range=[{min(loo)},{max(loo)}]")
    for k, c in sorted(cells.items()):
        print(f"  {k:30s} AUC={c['auc']:.3f}  OOD-FPR={c['ood_fpr']:.3f}  "
              f"CI{c['ci']}  oriented={c['sign_oriented']}")
    print(f"\nwritten: {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
