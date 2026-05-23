#!/usr/bin/env python
"""WS-A: balanced, calibrated & conformal re-analysis of Latent Sentinel probes.

EMNLP 2026 revision. See paper/EMNLP2026_STRATEGY.md (Pillars 1 & 4) and
EMNLP2026_PLAN.md (WS-A). Pure re-analysis of per-sample scores ALREADY on disk —
NO re-inference, NO notebook surgery.

Sources auto-detected:
  * pkl: data/detailed_evaluation_results*.pkl
      {true_labels, label_map={1:'Benign',0:'Jailbreak/Redteam'}, model_name,
       predictions_by_layer={'{N}_layers':{'logits':(n,2)}}}.  Column j == class j,
       so P(jailbreak)=softmax(logits)[:,0].
  * predictions dir: data/predictions/<model_dir>/(*)_<N>_predictions.csv
      cols [text,label,pred,prob_benign,logit0,logit1]; label 1=benign 0=jailbreak;
      cross-layer row-aligned; P(jailbreak)=1-prob_benign.

Positive class for detection = jailbreak (label 0). We report what the paper hid:
ROC/PR-AUC, F1, benign specificity, FPR@95TPR, TPR@1%FPR, ECE — at the paper's
default tau=0.5, at a balanced-accuracy tau picked on a frozen calibration split,
and at a CONFORMAL tau with a finite-sample guarantee that benign FPR <= alpha.

Env: pinned in requirements.txt (see `bootstrap-latents` conda env)
  python ws_a_balanced_metrics.py --source <pkl|dir> [...] --out ../../data/emnlp2026/ws_a
  python ws_a_balanced_metrics.py --all   # all known sources -> consolidated table
"""
from __future__ import annotations
import argparse, glob, json, os, pickle, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_curve, brier_score_loss)

REPO = Path(__file__).resolve().parents[2]
KNOWN_SOURCES = [
    REPO / "data/detailed_evaluation_results.pkl",                       # Llama-3.1-8B_JB
    REPO / "data/detailed_evaluation_results_predefined_threshold.pkl",  # Qwen2.5-7B aligned
    REPO / "data/detailed_mlp_evaluation_results.pkl",                   # Qwen2.5-7B MLP probe
    REPO / "data/predictions/stander_jailbreak_eval",                    # Llama-3.1-8B aligned
    REPO / "data/predictions/stander_jailbreak_eval_jailbreak",          # Llama-3.1-8B_JB
    REPO / "data/predictions/stander_jailbreak_eval_qwen",               # Qwen2.5-7B aligned
    REPO / "data/predictions/stander_jailbreak_eval_qwen_jailbreak",     # Qwen2.5-7B_JB
]
DIR_MODEL = {
    "stander_jailbreak_eval": "Llama-3.1-8B-Instruct (aligned)",
    "stander_jailbreak_eval_jailbreak": "Llama-3.1-8B-Instruct_JB",
    "stander_jailbreak_eval_qwen": "Qwen2.5-7B-Instruct (aligned)",
    "stander_jailbreak_eval_qwen_jailbreak": "Qwen2.5-7B-Instruct_JB",
}


# ---------- score loading (normalize everything to: y, P[n,L], layers, model) ----------
def _softmax_jb(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True))[:, 0]  # class 0 == jailbreak


def load_source(path: Path):
    if path.is_file() and path.suffix == ".pkl":
        d = pickle.loads(path.read_bytes())
        true = np.asarray(d["true_labels"]).astype(int)
        lm = {int(k): v for k, v in d["label_map"].items()}
        jb = next(k for k, v in lm.items() if "Jail" in v or "Red" in v)
        y = (true == jb).astype(int)
        pbl = d["predictions_by_layer"]
        layers = sorted(pbl, key=lambda s: int(s.split("_")[0]))
        P = np.column_stack([_softmax_jb(np.asarray(pbl[L]["logits"], float)) for L in layers])
        return y, P, [int(L.split("_")[0]) for L in layers], d.get("model_name", path.stem)
    if path.is_dir():
        files = []
        for c in glob.glob(str(path / "*_predictions.csv")):
            m = re.search(r"_(\d+)_predictions\.csv$", os.path.basename(c))
            if m:
                files.append((int(m.group(1)), c))
        files.sort()
        base = pd.read_csv(files[0][1])
        y = (base["label"].values == 0).astype(int)            # jailbreak(0) = positive
        P = np.column_stack([1.0 - pd.read_csv(c)["prob_benign"].values for _, c in files])
        return y, P, [n for n, _ in files], DIR_MODEL.get(path.name, path.name)
    raise ValueError(f"unrecognized source: {path}")


# ---------- metrics ----------
def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def fpr_at_tpr(y, p, t):
    fpr, tpr, _ = roc_curve(y, p)
    return float(fpr[min(np.searchsorted(tpr, t), len(fpr) - 1)])


def tpr_at_fpr(y, p, t):
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= t
    return float(tpr[ok].max()) if ok.any() else 0.0


def op(y, p, tau):
    pred = (p >= tau).astype(int)
    b, j = y == 0, y == 1
    return dict(threshold=round(float(tau), 4),
                accuracy=round(float((pred == y).mean()), 4),
                precision_jb=round(float(precision_score(y, pred, zero_division=0)), 4),
                recall_jb=round(float(recall_score(y, pred, zero_division=0)), 4),
                f1_jb=round(float(f1_score(y, pred, zero_division=0)), 4),
                benign_specificity=round(float((pred[b] == 0).mean()), 4),
                benign_fpr=round(float((pred[b] == 1).mean()), 4),
                balanced_accuracy=round(float(0.5 * ((pred[j] == 1).mean()
                                                     + (pred[b] == 0).mean())), 4))


def conformal_tau(s_benign_cal: np.ndarray, alpha: float) -> float:
    """Split-conformal one-sided threshold s.t. P(flag | benign) <= alpha with
    finite-sample validity. Predict jailbreak iff score >= tau; tau is the
    ceil((n+1)(1-alpha))/n empirical quantile of benign calibration scores."""
    s = np.sort(s_benign_cal)
    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float(np.nextafter(s[-1], np.inf))  # cannot guarantee; abstain-high
    return float(s[k - 1])


def analyze(y, p, cal, tst):
    yt, pt = y[tst], p[tst]
    yc, pc = y[cal], p[cal]
    grid = np.unique(np.quantile(pc, np.linspace(0, 1, 201)))
    bal = [0.5 * (((pc >= t)[yc == 1]).mean() + ((pc < t)[yc == 0]).mean()) for t in grid]
    tau_bal = float(grid[int(np.argmax(bal))])
    conf = {}
    for a in (0.01, 0.05, 0.10):
        tau = conformal_tau(pc[yc == 0], a)
        m = op(yt, pt, tau)
        conf[f"alpha_{a}"] = dict(tau=round(tau, 4),
                                  test_benign_fpr=m["benign_fpr"],
                                  test_recall_jb=m["recall_jb"],
                                  test_balanced_acc=m["balanced_accuracy"])
    yb = (yt == 1).astype(int)
    return dict(n_test=int(len(tst)),
                roc_auc=round(float(roc_auc_score(yt, pt)), 4),
                pr_auc_jb=round(float(average_precision_score(yt, pt)), 4),
                brier=round(float(brier_score_loss(yb, pt)), 4),
                ece=round(ece(pt, yb), 4),
                fpr_at_95tpr=round(fpr_at_tpr(yb, pt, 0.95), 4),
                tpr_at_1pct_fpr=round(tpr_at_fpr(yb, pt, 0.01), 4),
                op_default_0p5=op(yt, pt, 0.5),
                op_balanced_cal=op(yt, pt, tau_bal),
                conformal=conf)


def run_source(path: Path, seed: int, cal_frac: float, out: Path) -> dict:
    y, P, layers, model = load_source(path)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    cal, tst = [], []
    for c in (0, 1):
        ci = idx[y == c]
        rng.shuffle(ci)
        k = int(round(len(ci) * cal_frac))
        cal += list(ci[:k])
        tst += list(ci[k:])
    cal, tst = np.array(sorted(cal)), np.array(sorted(tst))

    per_layer = {f"{L}": analyze(y, P[:, i], cal, tst) for i, L in enumerate(layers)}
    mean_p = P.mean(1)
    vote_p = (P >= 0.5).mean(1)
    cal_auc = np.array([roc_auc_score(y[cal], P[cal, i]) for i in range(P.shape[1])])
    topk = np.argsort(-cal_auc)[:5]
    ens = {"mean_prob": analyze(y, mean_p, cal, tst),
           "majority_vote": analyze(y, vote_p, cal, tst),
           "top5_mean_by_calAUC": analyze(y, P[:, topk].mean(1), cal, tst)}
    # AUDIT FIX C1: select the reported layer by CALIBRATION AUC (label-blind to
    # the test split), NOT by test AUC. Test-AUC selection was optimistic model
    # selection and propagated a test-cherry-picked layer downstream.
    best_i = int(np.argmax(cal_auc))
    best = (f"{layers[best_i]}", per_layer[f"{layers[best_i]}"])
    # also expose the test-AUC argmax purely for transparency (NOT reported)
    best_test = max(per_layer.items(), key=lambda kv: kv[1]["roc_auc"])
    res = dict(source=str(path), model=model, n=int(len(y)),
               n_benign=int((y == 0).sum()), n_jailbreak=int((y == 1).sum()),
               seed=seed, cal_frac=cal_frac, n_cal=int(len(cal)), n_test=int(len(tst)),
               layers=layers,
               best_layer_by_auc=dict(layer=best[0], selected_by="calibration_auc",
                                      cal_auc=round(float(cal_auc[best_i]), 4),
                                      **best[1]),
               best_layer_by_TEST_auc_transparency_only=best_test[0],
               per_layer=per_layer, ensembles=ens)
    out.mkdir(parents=True, exist_ok=True)
    tag = path.stem if path.is_file() else path.name
    (out / f"{tag}__balanced.json").write_text(json.dumps(res, indent=2))
    return res


def consolidated_row(r: dict) -> dict:
    b = r["best_layer_by_auc"]
    c5 = b["conformal"]["alpha_0.05"]
    return {
        "model": r["model"], "n": r["n"], "best_layer": b["layer"],
        "roc_auc": b["roc_auc"], "ece": b["ece"], "fpr@95tpr": b["fpr_at_95tpr"],
        "DEF_recall": b["op_default_0p5"]["recall_jb"],
        "DEF_benign_spec": b["op_default_0p5"]["benign_specificity"],
        "DEF_bal_acc": b["op_default_0p5"]["balanced_accuracy"],
        "BALcal_recall": b["op_balanced_cal"]["recall_jb"],
        "BALcal_benign_spec": b["op_balanced_cal"]["benign_specificity"],
        "BALcal_bal_acc": b["op_balanced_cal"]["balanced_accuracy"],
        "CONF@a=.05_test_benign_fpr": c5["test_benign_fpr"],
        "CONF@a=.05_test_recall": c5["test_recall_jb"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append", default=[], type=Path)
    ap.add_argument("--all", action="store_true", help="process all KNOWN_SOURCES")
    ap.add_argument("--out", type=Path, default=REPO / "data/emnlp2026/ws_a")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cal-frac", type=float, default=0.5)
    a = ap.parse_args()
    srcs = list(a.source)
    if a.all:
        srcs = [p for p in KNOWN_SOURCES if p.exists()]
    if not srcs:
        ap.error("give --source <path> (repeatable) or --all")

    rows = []
    for s in srcs:
        try:
            r = run_source(s, a.seed, a.cal_frac, a.out)
        except Exception as e:  # noqa: BLE001 - keep batch going, surface which source
            print(f"[skip] {s}: {type(e).__name__}: {e}")
            continue
        rows.append(consolidated_row(r))
        b = r["best_layer_by_auc"]
        print(f"\n== {r['model']}  (n={r['n']}, src={Path(r['source']).name})")
        print(f"   best L{b['layer']}  AUC={b['roc_auc']}  ECE={b['ece']}  "
              f"FPR@95TPR={b['fpr_at_95tpr']}")
        print(f"   default tau=0.5 : recall={b['op_default_0p5']['recall_jb']}  "
              f"benign_spec={b['op_default_0p5']['benign_specificity']}  "
              f"bal_acc={b['op_default_0p5']['balanced_accuracy']}")
        cf = b["conformal"]["alpha_0.05"]
        print(f"   conformal a=.05 : tau={cf['tau']}  test_benign_FPR={cf['test_benign_fpr']}"
              f"  test_recall={cf['test_recall_jb']}  (guarantee: FPR<=0.05)")
    if rows:
        df = pd.DataFrame(rows)
        a.out.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out / "CONSOLIDATED_corrected_metrics.csv", index=False)
        print("\n" + "=" * 78 + "\nCONSOLIDATED (best layer per model)\n" + "=" * 78)
        print(df.to_string(index=False))
        print(f"\nwritten: {a.out}/CONSOLIDATED_corrected_metrics.csv  (+ per-source JSON)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
