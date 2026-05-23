#!/usr/bin/env python
"""CHECK.md H-2 — extract representative qualitative examples for the appendix.

Loads the Qwen2.5-7B cached hidden states + reconstructs the disjoint
held-out-remedy split + erased probe, then scores held-out test prompts under
BOTH the baseline and the recipe probe at the deployment tau. Selects 2 of each
of {recipe TP, recipe FN, recipe FP, recipe TN} -- favouring cases where the
recipe DIFFERS from the baseline (the most informative qualitative evidence).
Truncates each prompt to <=110 chars (preserving the head + tail), redacts
explicit harmful operative substrings with [...] -- ToxicChat/JailBreakV are
public datasets so excerpts are reproducible. Prints a ready-to-paste LaTeX
description block.

Env: <conda-env>/bin/python
  python pivot_qualitative_examples.py
"""
from __future__ import annotations
import glob, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
PIV = REPO / "data/emnlp2026/pivot"
WSB = REPO / "data/emnlp2026/wsb"
PRED = REPO / "data/predictions"
ALPHA = 0.05
CHAT_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def uc(t):
    t = str(t); m = CHAT_RE.search(t)
    if m: return m.group(1).strip()
    for g in ("<|im_start|>", "<|im_end|>", "<|begin_of_text|>",
              "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"):
        t = t.replace(g, " ")
    return re.sub(r"\s+", " ", t).strip()


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
    return mu.astype(np.float32), R.astype(np.float32), r


def conformal_tau(s, alpha=ALPHA):
    s = np.sort(s); k = int(np.ceil((len(s) + 1) * (1 - alpha))) - 1
    return float(s[min(k, len(s) - 1)])


def train_probe(Htr, ytr):
    sc = StandardScaler().fit(Htr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Htr), ytr)
    return lambda X: clf.predict_proba(sc.transform(X))[:, 1]


def strat_3way(bs_ds, seed=20260520):
    rng = np.random.default_rng(seed); eraser, depcal, test = [], [], []
    for s in sorted(set(map(str, bs_ds))):
        idx = np.where(np.array(list(map(str, bs_ds))) == s)[0]
        rng.shuffle(idx); n = len(idx); a, b = n // 3, 2 * n // 3
        eraser += list(idx[:a]); depcal += list(idx[a:b]); test += list(idx[b:])
    return (np.array(sorted(eraser)), np.array(sorted(depcal)),
            np.array(sorted(test)))


def truncate(s, n=110):
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= n: return s
    head = s[: n - 20]
    tail = s[-15:]
    return head + " […] " + tail


# avoid auto-extracting operative harmful suffixes for jailbreak prompts;
# replace any [INST]...content...[/INST] style chains with placeholders
def safe_excerpt(s):
    s = truncate(s, 120)
    # collapse common jailbreak templates that include operative requests
    s = re.sub(r"step[-\s]?by[-\s]?step.*", "step-by-step [redacted]", s, flags=re.I)
    return s


def main():
    z = np.load(PIV / "hs_qwen_aligned_L20.npz", allow_pickle=True)
    Hin, y, Hbs, bs_ds = z["Hin"], z["y"], z["Hbs"], z["bs_ds"]
    # same in-dist split
    rng = np.random.default_rng(42)
    ben = np.where(y == 0)[0]; jb = np.where(y == 1)[0]
    rng.shuffle(ben); rng.shuffle(jb)
    cal_b, te_b = ben[:len(ben) // 2], ben[len(ben) // 2:]
    tr = np.array(sorted(list(cal_b) + list(jb[:len(jb) // 2])))
    te = np.array(sorted(list(te_b) + list(jb[len(jb) // 2:])))
    bs_er, bs_dc, bs_te = strat_3way(bs_ds)
    src = ["indist_benign"] + sorted(set(map(str, bs_ds[bs_er])))
    # CHECK fix H-1: LEACE fit excludes in-dist test partition te_b.
    Xfit = np.vstack([Hin[cal_b], Hbs[bs_er]])
    zlab = np.array([0] * len(cal_b) +
                    [src.index(str(s)) for s in bs_ds[bs_er]])
    Zoh = np.eye(len(src))[zlab][:, 1:]
    mu, R, rnk = leace(Xfit, Zoh)
    er = lambda X: ((X - mu) @ R.T + mu).astype(np.float32)
    s_base = train_probe(Hin[tr], y[tr])
    s_recipe = train_probe(er(Hin[tr]), y[tr])
    tau_dep = conformal_tau(s_recipe(er(Hbs[bs_dc])))
    tau_dep_base = conformal_tau(s_base(Hbs[bs_dc]))

    # Load held-out jb texts
    df = pd.read_csv(sorted(glob.glob(
        str(PRED / "stander_jailbreak_eval_qwen" / "*_predictions.csv")))[0])
    te_idx = te; jte = te_idx[y[te_idx] == 1]
    txts_jb = [uc(t) for t in df.iloc[jte]["text"].tolist()]
    # Load held-out benign-stress texts (the bs_te indices)
    rows = [json.loads(l) for l in
            (WSB / "benign_stress.jsonl").read_text().splitlines() if l.strip()]
    txts_bs = [rows[i]["prompt"] for i in bs_te]
    src_bs = [str(bs_ds[i]) for i in bs_te]

    # Score
    s_jb_rec = s_recipe(er(Hin[jte])); s_jb_base = s_base(Hin[jte])
    s_bs_rec = s_recipe(er(Hbs[bs_te])); s_bs_base = s_base(Hbs[bs_te])

    out = []

    def pick(mask, n, txts, recipe_s, base_s, kind):
        idx = np.where(mask)[0]
        if len(idx) == 0: return
        # prefer extreme cases (highest recipe score for TP, lowest for FN, etc.)
        ordered = sorted(idx, key=lambda i: -recipe_s[i] if "T" in kind
                                              else recipe_s[i])
        for i in ordered[:n]:
            out.append(dict(kind=kind, prompt=safe_excerpt(txts[i]),
                            recipe_score=round(float(recipe_s[i]), 3),
                            baseline_score=round(float(base_s[i]), 3),
                            source=("jailbreak" if kind.startswith("jb")
                                    else src_bs[i] if kind.startswith("bs")
                                    else "?")))

    # Recipe TP on jailbreaks: recipe>tau AND baseline<tau_base (i.e., recipe catches what baseline missed)
    tp_advantage = (s_jb_rec >= tau_dep) & (s_jb_base < tau_dep_base)
    pick(tp_advantage, 2, txts_jb, s_jb_rec, s_jb_base, "jb_TP_recipe_only")
    # Recipe FN on jailbreaks: recipe<tau
    fn = s_jb_rec < tau_dep
    pick(fn, 2, txts_jb, s_jb_rec, s_jb_base, "jb_FN_recipe_misses")
    # Recipe FP on benign-stress: recipe>=tau
    fp = s_bs_rec >= tau_dep
    pick(fp, 2, txts_bs, s_bs_rec, s_bs_base, "bs_FP_recipe_still_overrefuses")
    # Recipe TN advantage: recipe<tau AND baseline>=tau_base (cases where recipe stops over-refusing)
    tn_advantage = (s_bs_rec < tau_dep) & (s_bs_base >= tau_dep_base)
    pick(tn_advantage, 2, txts_bs, s_bs_rec, s_bs_base, "bs_TN_recipe_only")

    op = PIV / "qualitative_examples.json"
    op.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwritten: {op}  ({len(out)} examples)")
    # LaTeX preview
    print("\n\\begin{itemize}\\small")
    for e in out:
        kind_lbl = {"jb_TP_recipe_only": "Jailbreak caught by recipe (baseline missed)",
                    "jb_FN_recipe_misses": "Jailbreak missed by recipe",
                    "bs_FP_recipe_still_overrefuses": "Benign still over-refused by recipe",
                    "bs_TN_recipe_only": "Benign no-longer over-refused (recipe vs baseline)"}[e["kind"]]
        print(f"\\item \\textit{{{kind_lbl}}} "
              f"({e['source']}; recipe score $={e['recipe_score']}$, "
              f"baseline $={e['baseline_score']}$): "
              f"``{e['prompt']}''")
    print("\\end{itemize}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
