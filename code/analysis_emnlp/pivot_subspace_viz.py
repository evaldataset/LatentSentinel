#!/usr/bin/env python
"""B5 + B3 — visualize the rank-3 register subspace + per-dimension interpretation.

(B5) For each backbone, project hidden states onto the 2 leading register
directions, scatter by source. Strong visual evidence that the rank-3
subspace separates the 4 benign sources + jailbreak cluster.

(B3) For each of the 3 register directions, report which surface feature
(length / question-mark count / imperative-onset / role-token presence /
code-marker presence) correlates most strongly with that direction's
activation across the held-out benign prompts. This gives an interpretable
account of what the rank-3 subspace encodes.

Output: paper/fig_rank3_register.pdf (Qwen as representative; appendix figure)
        + data/emnlp2026/pivot/rank3_interpretation.json
CPU only.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
PIV = REPO / "data/emnlp2026/pivot"
WSB = REPO / "data/emnlp2026/wsb"
PRED = REPO / "data/predictions"
CACHES = {"qwen_aligned": "hs_qwen_aligned_L20.npz",
          "llama_aligned": "hs_llama_aligned_L16.npz",
          "mistral_aligned": "hs_mistral_aligned_L16.npz"}
CHAT_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def uc(t):
    t = str(t); m = CHAT_RE.search(t)
    if m: return m.group(1).strip()
    for g in ("<|im_start|>", "<|im_end|>", "<|begin_of_text|>",
              "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"):
        t = t.replace(g, " ")
    return re.sub(r"\s+", " ", t).strip()


def fit_register_basis(Hin, y, Hbs, bs_ds):
    ben = np.where(y == 0)[0]
    sources = sorted(set(map(str, bs_ds)))
    src = ["indist_benign"] + sources
    Xfit = np.vstack([Hin[ben], Hbs])
    zlab = np.array([0] * len(ben) + [src.index(str(s)) for s in bs_ds])
    Zoh = np.eye(len(src))[zlab][:, 1:]
    mu = Xfit.mean(0); Xc = Xfit - mu; d = Xc.shape[1]
    eps = 1e-3
    Sig = (Xc.T @ Xc) / len(Xc) + eps * np.eye(d)
    w, U = np.linalg.eigh(Sig); w = np.clip(w, eps, None)
    isq = (U * w ** -0.5) @ U.T
    Zc = Zoh - Zoh.mean(0)
    M = isq @ ((Xc.T @ Zc) / len(Xc))
    Um, sm, _ = np.linalg.svd(M, full_matrices=False)
    r = max(int((sm > 1e-6).sum()), 1)
    return mu.astype(np.float32), Um[:, :r].astype(np.float32), r


def surface_features(text):
    """Quick register-style features."""
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


def main():
    # ---- B5: scatter plot for Qwen (representative) ----
    z = np.load(PIV / CACHES["qwen_aligned"], allow_pickle=True)
    Hin, y, Hbs, bs_ds = z["Hin"], z["y"], z["Hbs"], z["bs_ds"]
    mu, Q, rnk = fit_register_basis(Hin, y, Hbs, bs_ds)

    # project everything onto Q[:,:2] (first 2 register dims)
    P_in = (Hin - mu) @ Q[:, :2]
    P_bs = (Hbs - mu) @ Q[:, :2]
    bs_str = np.array(list(map(str, bs_ds)))

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    SUB = 1500  # subsample for clarity
    rng = np.random.default_rng(0)
    def sub(arr):
        if len(arr) <= SUB: return arr
        idx = rng.choice(len(arr), SUB, replace=False)
        return arr[idx]
    ben = np.where(y == 0)[0]; jb = np.where(y == 1)[0]
    pts_in_benign = sub(P_in[ben])
    pts_in_jb     = sub(P_in[jb])
    ax.scatter(pts_in_benign[:, 0], pts_in_benign[:, 1],
               s=4, alpha=0.35, color="#888", label="In-dist benign (Alpaca-style)")
    for src, color, lab in [("orbench_hard",      "#1f77b4", "OR-Bench-Hard"),
                             ("toxicchat_benign", "#2ca02c", "ToxicChat-benign"),
                             ("xstest_safe",      "#ff7f0e", "XSTest-safe")]:
        mask = bs_str == src
        if mask.any():
            p = sub(P_bs[mask])
            ax.scatter(p[:, 0], p[:, 1], s=5, alpha=0.5, color=color, label=lab)
    ax.scatter(pts_in_jb[:, 0], pts_in_jb[:, 1],
               s=4, alpha=0.4, color="#d62728", marker="x",
               label="Jailbreak (JailBreakV)")
    ax.set_xlabel("Register direction 1", fontsize=9)
    ax.set_ylabel("Register direction 2", fontsize=9)
    ax.set_title("Rank-3 benign-register subspace (Qwen2.5-7B, L20)\n"
                 "first 2 of 3 LEACE source-id directions",
                 fontsize=9)
    ax.legend(loc="best", fontsize=7, frameon=True)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    out_pdf = REPO / "paper/fig_rank3_register.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"written: {out_pdf}")

    # ---- B3: per-dimension interpretation ----
    # Project hidden states onto each of the 3 register directions; correlate
    # with surface features over the held-out diverse benign + in-dist benign.
    import pandas as pd
    # use a sample of in-dist benign + all bs for prompts
    df = pd.read_csv(sorted((PRED / "stander_jailbreak_eval_qwen").glob(
        "*_predictions.csv"))[0])
    prompts_in = [uc(t) for t in df["text"].tolist()]
    rows = [json.loads(l) for l in
            (WSB / "benign_stress.jsonl").read_text().splitlines() if l.strip()]
    prompts_bs = [r["prompt"] for r in rows]
    # features for all
    F_in = pd.DataFrame([surface_features(p) for p in prompts_in])
    F_bs = pd.DataFrame([surface_features(p) for p in prompts_bs])
    # project hidden states onto each of 3 register dims
    proj_in = (Hin - mu) @ Q                       # (N_in, 3)
    proj_bs = (Hbs - mu) @ Q                       # (N_bs, 3)
    interp = {}
    feats = F_in.columns.tolist()
    for k in range(min(3, Q.shape[1])):
        all_proj = np.concatenate([proj_in[:, k], proj_bs[:, k]])
        all_feat = pd.concat([F_in, F_bs], ignore_index=True)
        corrs = {}
        for f in feats:
            v = all_feat[f].values.astype(float)
            if v.std() > 0:
                corrs[f] = float(np.corrcoef(v, all_proj)[0, 1])
        sorted_c = sorted(corrs.items(), key=lambda kv: -abs(kv[1]))[:4]
        interp[f"dim_{k+1}"] = sorted_c
        print(f"dim {k+1} top corr: {sorted_c}")
    op = PIV / "rank3_interpretation.json"
    op.write_text(json.dumps(interp, indent=1))
    print(f"\nwritten: {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
