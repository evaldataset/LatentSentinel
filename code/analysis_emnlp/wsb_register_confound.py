#!/usr/bin/env python
"""WS-B (no-GPU): does the probe detect adversarial INTENT or benign REGISTER?

EMNLP 2026, Strategy Pillar 2 (mechanism of failure). Uses ONLY the prompt text +
per-layer probe scores already in data/predictions/<model>/*_predictions.csv
(2000 benign + 2000 jailbreak, with text). No model, no GPU, no re-inference.

Decisive tests:
  (T1) Surface-register-only logistic regression (cheap hand features, NO model) ->
       true label. If high AUC, "jailbreak vs benign" is largely a register artifact
       of the JailbreakV-vs-Alpaca construction (cf. "When Benchmarks Lie" 2602.14161,
       "False Sense of Security" 2509.03888).
  (T2) Surface-register-only LR -> the PROBE's own decision. High AUC => the probe is
       (largely) reconstructible from register alone, i.e. not reading intent.
  (T3) Among benign prompts only: which surface features predict a benign prompt being
       flagged (false positive)? Point-biserial corr of each feature with probe score.
  (T4) Probe-score vs prompt length / instruction-ness, within each class.

Env: pinned in requirements.txt (see `bootstrap-latents` conda env)
  python wsb_register_confound.py --model-dir stander_jailbreak_eval_qwen --layer 20
  python wsb_register_confound.py --all-best   # the WS-A best layer per model
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

REPO = Path(__file__).resolve().parents[2]
PRED = REPO / "data/predictions"
# WS-A best layer per model (from CONSOLIDATED_corrected_metrics.csv)
BEST = {"stander_jailbreak_eval": 16, "stander_jailbreak_eval_jailbreak": 15,
        "stander_jailbreak_eval_qwen": 20, "stander_jailbreak_eval_qwen_jailbreak": 16}
NAME = {"stander_jailbreak_eval": "Llama-3.1-8B (aligned)",
        "stander_jailbreak_eval_jailbreak": "Llama-3.1-8B_JB",
        "stander_jailbreak_eval_qwen": "Qwen2.5-7B (aligned)",
        "stander_jailbreak_eval_qwen_jailbreak": "Qwen2.5-7B_JB"}

CHAT_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
IMP_VERBS = ("write", "explain", "describe", "list", "give", "provide", "create",
             "generate", "tell", "show", "make", "develop", "design", "act as",
             "pretend", "imagine", "roleplay", "you are", "ignore", "bypass")
ROLE_TOK = ("dan", "do anything now", "jailbreak", "developer mode", "persona",
            "hacker", "no restrictions", "without any", "unfiltered", "as an ai")


def user_content(t: str) -> str:
    t = str(t)
    m = CHAT_RE.search(t)
    if m:
        return m.group(1).strip()
    for tag in ("<|im_start|>", "<|im_end|>", "<|begin_of_text|>",
                "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"):
        t = t.replace(tag, " ")
    return re.sub(r"\s+", " ", t).strip()


def feats(s: str) -> dict:
    s = s or ""
    low = s.lower()
    w = s.split()
    nw = max(len(w), 1)
    return {
        "char_len": len(s),
        "word_len": len(w),
        "log_len": np.log1p(len(s)),
        "avg_word_len": float(np.mean([len(x) for x in w])) if w else 0.0,
        "n_question": s.count("?"),
        "n_newline": s.count("\n"),
        "frac_upper": sum(c.isupper() for c in s) / max(len(s), 1),
        "n_digits": sum(c.isdigit() for c in s),
        "has_code": int(any(k in low for k in ("```", "def ", "import ", "<div", "</", "{", "};"))),
        "imp_verb_frac": sum(low.count(v) for v in IMP_VERBS) / nw,
        "starts_imperative": int(low.split()[0] in IMP_VERBS) if w else 0,
        "role_token": int(any(k in low for k in ROLE_TOK)),
        "first_person": low.count(" i ") + low.startswith("i "),
        "second_person": low.count("you "),
        "polite": int(any(k in low for k in ("please", "could you", "thank"))),
    }


def load(model_dir: str):
    d = PRED / model_dir
    files = []
    for c in glob.glob(str(d / "*_predictions.csv")):
        m = re.search(r"_(\d+)_predictions\.csv$", os.path.basename(c))
        if m:
            files.append((int(m.group(1)), c))
    files.sort()
    base = pd.read_csv(files[0][1])
    y_jb = (base["label"].values == 0).astype(int)            # 1 = jailbreak (positive)
    P = {n: 1.0 - pd.read_csv(c)["prob_benign"].values for n, c in files}
    txt = [user_content(t) for t in base["text"].tolist()]
    return base, y_jb, P, txt, [n for n, _ in files]


def lr_auc(X, target, seed):
    """5-fold CV AUC of an L2 logistic regression on standardized features.
    AUDIT FIX M5: StratifiedKFold(shuffle, random_state=seed) — the data are
    concatenated benign-then-jailbreak, so the previous default cv=5 (unshuffled,
    unstratified KFold) produced near-single-class folds and silently ignored
    the seed."""
    from sklearn.model_selection import StratifiedKFold
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=int(seed))
    proba = cross_val_predict(pipe, X, target, cv=cv, method="predict_proba",
                              n_jobs=4)[:, 1]
    return float(roc_auc_score(target, proba)), proba


def analyze(model_dir: str, layer: int, seed: int) -> dict:
    base, y_jb, P, txt, layers = load(model_dir)
    if layer not in P:
        layer = min(P, key=lambda L: abs(L - layer))
    s = P[layer]                       # jailbreak score, best layer
    s_mean = np.mean(np.column_stack(list(P.values())), axis=1)
    F = pd.DataFrame([feats(t) for t in txt])
    cols = list(F.columns)
    X = F.values

    rng = np.random.default_rng(seed)
    _ = rng  # CV determinism now via StratifiedKFold(random_state=seed) in lr_auc

    # T1: register-only -> true label
    auc_label, _ = lr_auc(X, y_jb, seed)
    # T2: register-only -> probe decision (best layer, default 0.5 and argmax-of-score)
    probe_pred = (s >= 0.5).astype(int)
    if probe_pred.std() == 0:                      # degenerate at 0.5 -> use median split
        probe_pred = (s >= np.median(s)).astype(int)
    auc_probe, _ = lr_auc(X, probe_pred, seed)
    # how often does register-only LR agree with the probe vs with truth
    agree_probe_truth = float((probe_pred == y_jb).mean())

    # T3: benign-only — feature corr with probe score (point-biserial via Pearson)
    bmask = y_jb == 0
    sb = s[bmask]
    Fb = F[bmask]
    corr_benign = {c: float(np.corrcoef(Fb[c].values, sb)[0, 1])
                   if Fb[c].std() > 0 else 0.0 for c in cols}
    corr_benign = dict(sorted(corr_benign.items(),
                              key=lambda kv: -abs(kv[1])))
    # flagged-benign vs clean-benign feature gaps (standardized mean diff)
    thr = np.quantile(s, 0.5)
    flagged = bmask & (s >= thr)
    clean = bmask & (s < thr)
    smd = {}
    for c in cols:
        a, b = F[c][flagged].values, F[c][clean].values
        sd = np.sqrt(0.5 * (a.var() + b.var())) or 1.0
        smd[c] = float((a.mean() - b.mean()) / sd)
    smd = dict(sorted(smd.items(), key=lambda kv: -abs(kv[1]))[:8])

    # T4: within-class corr(prob, length)
    jmask = y_jb == 1
    def pc(mask, key):
        v = F[key][mask].values
        return float(np.corrcoef(v, s[mask])[0, 1]) if v.std() > 0 else 0.0

    return {
        "model": NAME.get(model_dir, model_dir), "layer_used": int(layer),
        "n": int(len(y_jb)), "n_benign": int(bmask.sum()),
        "probe_auc_truth_bestlayer": round(float(roc_auc_score(y_jb, s)), 4),
        "T1_register_only_AUC_vs_truth": round(auc_label, 4),
        "T2_register_only_AUC_vs_probe_decision": round(auc_probe, 4),
        "probe_acc_vs_truth_at_split": round(agree_probe_truth, 4),
        "T3_benign_only_top_feature_corr_with_score":
            {k: round(v, 3) for k, v in list(corr_benign.items())[:8]},
        "T3_flagged_vs_clean_benign_StdMeanDiff":
            {k: round(v, 3) for k, v in smd.items()},
        "T4_corr_len_score_benign": round(pc(bmask, "log_len"), 3),
        "T4_corr_len_score_jailbreak": round(pc(jmask, "log_len"), 3),
        "T4_corr_impverb_score_benign": round(pc(bmask, "imp_verb_frac"), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--all-best", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPO / "data/emnlp2026/wsb")
    a = ap.parse_args()
    jobs = ([(m, BEST[m]) for m in BEST] if a.all_best
            else [(a.model_dir, a.layer or BEST.get(a.model_dir, 15))])
    a.out.mkdir(parents=True, exist_ok=True)
    res = []
    for md, ly in jobs:
        r = analyze(md, ly, a.seed)
        res.append(r)
        print(f"\n== {r['model']}  (best L{r['layer_used']}, probe AUC vs truth "
              f"{r['probe_auc_truth_bestlayer']})")
        print(f"   T1 register-only LR  -> truth         AUC = "
              f"{r['T1_register_only_AUC_vs_truth']}")
        print(f"   T2 register-only LR  -> probe decision AUC = "
              f"{r['T2_register_only_AUC_vs_probe_decision']}")
        print(f"   T3 benign-only |corr| top: " + ", ".join(
            f"{k}={v:+.2f}" for k, v in
            list(r['T3_benign_only_top_feature_corr_with_score'].items())[:5]))
        print(f"   T4 corr(len,score): benign={r['T4_corr_len_score_benign']:+.2f} "
              f"jailbreak={r['T4_corr_len_score_jailbreak']:+.2f}")
    (a.out / "register_confound.json").write_text(json.dumps(res, indent=2))
    print(f"\nwritten: {a.out}/register_confound.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
