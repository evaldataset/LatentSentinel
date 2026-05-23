#!/usr/bin/env python
"""CHECK fix H-8 / G7 — Arditi 2024 refusal-direction probe baseline.

Arditi et al. (NeurIPS 2024, "Refusal in Language Models Is Mediated by a
Single Direction") is the canonical 2024 representation-engineering descendant
in the linear-probe-on-residual-stream lineage. The paper claims to characterize
this lineage but omits this baseline; we add it under the same uniform
split-conformal protocol used for the other six baselines.

Algorithm (Arditi et al. 2024):
  1. Fit a "refusal direction" d at the target layer:
       d = mean(h | harmful) - mean(h | harmless),  d /= ||d||
  2. Score new prompts by projection onto d:
       score(h) = h . d
  3. Higher score = more refusal-aligned = more "jailbreak detection".

For comparability with the other baselines we:
  - use the WS-A-selected layer per backbone (Qwen L20, Llama L16);
  - fit d on the CAL half of the in-distribution eval set
    (jailbreak = harmful, benign = harmless);
  - score the TEST half + the benign-stress set;
  - apply the same uniform split-conformal protocol (sign-orient by
    calibration AUC; tau at 5% benign-FPR on cal benign).

Inputs: cached `data/emnlp2026/pivot/hs_*.npz` (Hin, y, Hbs, bs_ds).
Output: appends Arditi rows to `data/emnlp2026/wsd/baselines.json`.

CPU only.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
PIV = REPO / "data/emnlp2026/pivot"
WSD = REPO / "data/emnlp2026/wsd"
ALPHA = 0.05
SEED = 42

CACHES = {"qwen_aligned": "hs_qwen_aligned_L20.npz",
          "llama_aligned": "hs_llama_aligned_L16.npz",
          "mistral_aligned": "hs_mistral_aligned_L16.npz"}


def conformal_tau(s, alpha=ALPHA):
    s = np.sort(s); n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(s[k]) if k < n else float(np.nextafter(s[-1], np.inf))


def split_idx(y, seed):
    rng = np.random.default_rng(seed)
    cal, tst = [], []
    for c in (0, 1):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        k = len(ci) // 2
        cal += list(ci[:k]); tst += list(ci[k:])
    return np.array(sorted(cal)), np.array(sorted(tst))


def fit_refusal_direction(H, y, cal_idx):
    """Difference-of-means direction at the target layer (Arditi 2024)."""
    cal_h = H[cal_idx]; cal_y = y[cal_idx]
    mu_harmful  = cal_h[cal_y == 1].mean(0)    # y==1 jailbreak
    mu_harmless = cal_h[cal_y == 0].mean(0)    # y==0 benign
    d = mu_harmful - mu_harmless
    return (d / (np.linalg.norm(d) + 1e-12)).astype(np.float32)


def main():
    out_all = {}
    for model, fn in CACHES.items():
        z = np.load(PIV / fn, allow_pickle=True)
        Hin, y, Hbs, bs_ds = z["Hin"], z["y"], z["Hbs"], z["bs_ds"]
        cal, tst = split_idx(y, SEED)

        # Arditi direction
        d = fit_refusal_direction(Hin, y, cal)
        det = Hin @ d                                   # (N,)
        det_bs = Hbs @ d                                # (M,)

        # sign orientation by calibration AUC
        auc_cal = roc_auc_score(y[cal], det[cal])
        if auc_cal < 0.5:
            det = -det; det_bs = -det_bs
            auc_cal = 1.0 - auc_cal

        # test AUC
        auc_test = float(roc_auc_score(y[tst], det[tst]))

        # uniform conformal protocol: tau on CAL benign at alpha=0.05
        ben_cal = det[cal][y[cal] == 0]
        tau = conformal_tau(ben_cal, ALPHA)

        # OOD FPR per source + overall
        bs_ds_str = np.array(list(map(str, bs_ds)))
        sources = sorted(set(bs_ds_str))
        per_source_fpr = {s: round(float((det_bs[bs_ds_str == s] >= tau).mean()), 4)
                          for s in sources}
        overall = float((det_bs >= tau).mean())

        # test benign FPR + recall
        tb = tst[y[tst] == 0]; tj = tst[y[tst] == 1]
        ind_fpr = float((det[tb] >= tau).mean())
        ind_recall = float((det[tj] >= tau).mean())

        res = {
            "model": model, "method": "arditi_refusal",
            "roc_auc_indist": round(auc_test, 4),
            "calibration_auc": round(float(auc_cal), 4),
            "tau_conformal_a0.05_on_indist_benign": round(tau, 6),
            "indist_benign_fpr_at_tau": round(ind_fpr, 4),
            "indist_jailbreak_recall_at_tau": round(ind_recall, 4),
            "benign_stress_FPR_at_tau": per_source_fpr,
            "benign_stress_FPR_overall": round(overall, 4),
        }
        out_all[model] = res
        print(f"\n== arditi_refusal / {model} ==")
        print(f"   in-dist test ROC-AUC = {res['roc_auc_indist']}")
        print(f"   @conf tau={res['tau_conformal_a0.05_on_indist_benign']}: "
              f"benign FPR={res['indist_benign_fpr_at_tau']} | "
              f"jb recall={res['indist_jailbreak_recall_at_tau']}")
        print(f"   benign-stress overall FPR @tau = {res['benign_stress_FPR_overall']} "
              f"(per source: {per_source_fpr})")

    # append to baselines.json
    out_path = WSD / "baselines.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    existing.setdefault("arditi_refusal", {})
    for m, r in out_all.items():
        existing["arditi_refusal"][m] = r
    out_path.write_text(json.dumps(existing, indent=1))
    print(f"\nappended to: {out_path}")

    # also write a standalone summary for convenience
    summ_path = WSD / "arditi_refusal_summary.json"
    summ_path.write_text(json.dumps(out_all, indent=1))
    print(f"written: {summ_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
