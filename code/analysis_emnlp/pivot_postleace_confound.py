#!/usr/bin/env python
"""CHECK fix H-6 — post-LEACE register-confound check.

The §Erase claim "a register-independent intent direction exists" can be
attacked by: LEACE removes linear concept-Z covariance; the genuine signal
might still be a *partial* register correlate, surviving LEACE because LEACE
only removes the fully-correlated component. The decisive test: re-run §5's
no-model surface classifier against the LEACE-ERASED probe's scores. If the
surface classifier still recovers ~0.85+ AUC against the erased probe's
decisions, the "register-independent" claim is broken. If it drops to ~0.5,
the causal claim is strengthened.

Procedure per backbone (CPU only, cached states):
  1. Reconstruct the leakage-free spine (matches `pivot_heldout_remedy.py`):
     in-dist split with seed 42, source-stratified 3-way split of Hbs,
     LEACE fit on Hin[cal_b] + Hbs[bs_er] (CHECK fix H-1).
  2. Score (a) baseline probe scores on full Hin and (b) erased probe scores
     on full Hin, both at the WS-A best layer.
  3. Build the same 6 surface features used in §5
     (length_chars, n_words, n_question, starts_imperative, has_role_token,
     has_code_marker) for each prompt from data/predictions/<dir>/*.csv.
  4. Fit an L2 logistic regression on the surface features against
     {true label, baseline probe decision, erased probe decision} with
     stratified 5-fold CV; report AUC.
  5. If AUC against {true label, baseline} stays high but against {erased}
     drops, the causal claim is reinforced.

Output: data/emnlp2026/pivot/postleace_confound.json
"""
from __future__ import annotations
import glob, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
PIV = REPO / "data/emnlp2026/pivot"
PRED = REPO / "data/predictions"
CACHES = {"qwen_aligned": ("hs_qwen_aligned_L20.npz", "stander_jailbreak_eval_qwen"),
          "llama_aligned": ("hs_llama_aligned_L16.npz", "stander_jailbreak_eval"),
          "mistral_aligned": ("hs_mistral_aligned_L16.npz", "stander_jailbreak_eval_qwen")}
CHAT_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def uc(t):
    t = str(t); m = CHAT_RE.search(t)
    if m: return m.group(1).strip()
    for g in ("<|im_start|>", "<|im_end|>", "<|begin_of_text|>",
              "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"):
        t = t.replace(g, " ")
    return re.sub(r"\s+", " ", t).strip()


def surface_features(text: str) -> dict:
    w = text.split()
    return dict(
        length_chars=len(text),
        n_words=len(w),
        n_question=text.count("?"),
        starts_imperative=1 if w and w[0].lower() in
            {"how", "tell", "give", "describe", "list", "guide", "help", "show",
             "make", "explain", "provide", "write"} else 0,
        has_role_token=1 if any(t in text.lower() for t in
            ("you are", "act as", "roleplay", "pretend", "imagine")) else 0,
        has_code_marker=1 if any(t in text for t in
            ("```", "def ", "import ", "function ", "class ")) else 0)


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
    R = sq @ (np.eye(d) - Q @ Q.T) @ isq
    return mu.astype(np.float32), R.astype(np.float32), int(r)


def strat_3way(bs_ds, seed=20260520):
    rng = np.random.default_rng(seed)
    eraser, depcal, test = [], [], []
    for s in sorted(set(map(str, bs_ds))):
        idx = np.where(np.array(list(map(str, bs_ds))) == s)[0]
        rng.shuffle(idx); n = len(idx); a, b = n // 3, 2 * n // 3
        eraser += list(idx[:a]); depcal += list(idx[a:b]); test += list(idx[b:])
    return (np.array(sorted(eraser)), np.array(sorted(depcal)),
            np.array(sorted(test)))


def train_probe(Htr, ytr):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Htr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Htr), ytr)
    return lambda X: clf.predict_proba(sc.transform(X))[:, 1]


def cv_auc(F: np.ndarray, target: np.ndarray, seed: int = 42) -> float:
    """L2 logreg on surface features vs target, 5-fold stratified CV AUC.
    `target` may be a continuous score (regress decisions) or label."""
    if len(set((target > target.mean()).astype(int))) < 2:
        return float("nan")
    y = (target > np.median(target)).astype(int)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(F, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(F[tr], y[tr])
        s = clf.predict_proba(F[te])[:, 1]
        aucs.append(float(roc_auc_score(y[te], s)))
    return float(np.mean(aucs))


def main():
    out = {}
    for model, (fn, pdir) in CACHES.items():
        print(f"\n========= {model} =========", flush=True)
        z = np.load(PIV / fn, allow_pickle=True)
        Hin, y, Hbs, bs_ds = z["Hin"], z["y"], z["Hbs"], z["bs_ds"]

        rng = np.random.default_rng(42)
        ben = np.where(y == 0)[0]; jb = np.where(y == 1)[0]
        rng.shuffle(ben); rng.shuffle(jb)
        cal_b, te_b = ben[:len(ben) // 2], ben[len(ben) // 2:]
        tr = np.array(sorted(list(cal_b) + list(jb[:len(jb) // 2])))
        bs_er, _, _ = strat_3way(bs_ds)
        src = ["indist_benign"] + sorted(set(map(str, bs_ds[bs_er])))
        # H-1 fix: LEACE fit on cal_b only
        Xfit = np.vstack([Hin[cal_b], Hbs[bs_er]])
        zlab = np.array([0] * len(cal_b) +
                        [src.index(str(s)) for s in bs_ds[bs_er]])
        Zoh = np.eye(len(src))[zlab][:, 1:]
        mu, R, rnk = leace(Xfit, Zoh)
        er = lambda X: ((X - mu) @ R.T + mu).astype(np.float32)

        # baseline & erased probe trained the same way as the spine
        score_base = train_probe(Hin[tr], y[tr])
        score_er   = train_probe(er(Hin[tr]), y[tr])

        # in-dist score arrays on the FULL in-dist eval set
        s_base = score_base(Hin)
        s_er   = score_er(er(Hin))

        # surface features per in-dist prompt
        df_path = sorted(glob.glob(str(PRED / pdir / "*_predictions.csv")))[0]
        df = pd.read_csv(df_path)
        prompts = [uc(t) for t in df["text"].tolist()]
        F = pd.DataFrame([surface_features(p) for p in prompts]).values.astype(float)

        # only use the test-time partition (so we don't leak the probe training set
        # into the surface-classifier-vs-decisions audit)
        te_b = np.array(sorted([i for i in range(len(y)) if i not in set(tr)]))

        out[model] = dict(
            leace_rank=rnk,
            n_eval=int(len(te_b)),
            auc_surface_vs_true_label=round(cv_auc(F[te_b], y[te_b].astype(float)), 4),
            auc_surface_vs_baseline_score=round(cv_auc(F[te_b], s_base[te_b]), 4),
            auc_surface_vs_erased_score=round(cv_auc(F[te_b], s_er[te_b]), 4),
            mean_score_base_benign=round(float(s_base[te_b][y[te_b] == 0].mean()), 4),
            mean_score_er_benign=round(float(s_er[te_b][y[te_b] == 0].mean()), 4),
            mean_score_base_jb=round(float(s_base[te_b][y[te_b] == 1].mean()), 4),
            mean_score_er_jb=round(float(s_er[te_b][y[te_b] == 1].mean()), 4),
        )
        r = out[model]
        print(f"  surface->true_label AUC = {r['auc_surface_vs_true_label']}")
        print(f"  surface->baseline_score AUC = {r['auc_surface_vs_baseline_score']}")
        print(f"  surface->erased_score   AUC = {r['auc_surface_vs_erased_score']}")
        print(f"  (drop = {r['auc_surface_vs_baseline_score'] - r['auc_surface_vs_erased_score']:.4f})")
    op = PIV / "postleace_confound.json"
    op.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
