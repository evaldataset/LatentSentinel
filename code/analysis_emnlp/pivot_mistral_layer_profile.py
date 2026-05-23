#!/usr/bin/env python
"""CHECK fix M-5 / G9 — Mistral-7B per-layer AUC + LEACE eigengap profile.

The paper claims Mistral-7B at L16 is a "held-out third family at mid-depth".
A reviewer will object that L16 is itself a hyperparameter borrowed from
Qwen+Llama (where it is the WS-A-selected layer). To defend the claim we
need Mistral's OWN per-layer profile:
  - calibration-AUC vs. layer 1-31 (probe trained per layer)
  - LEACE rank (source-id) eigengap vs. layer
showing L16 is mid-tier (not anomalously best) and rank-3 is the natural
choice on Mistral too.

Procedure:
  1. (GPU) Extract last-content-token hidden states for the in-dist
     4000-prompt eval set at every 4 layers (L4, L8, L12, L16, L20, L24, L28)
     using `unsloth/mistral-7b-instruct-v0.3` (ungated mirror).
  2. (CPU) Per layer, train a linear probe (LogReg) on 50/50 cal/tst split;
     report ROC-AUC on tst.
  3. (CPU) Per layer, fit LEACE on cal_benign+bs_eraser and report the
     source-ID singular-value spectrum (top 6); show where the eigengap is.
  4. Save `data/emnlp2026/pivot/mistral_layer_profile.json`.

Inputs we already have for free:
  - `data/emnlp2026/pivot/hs_mistral_aligned_L16.npz` (L16 only, full eval+bs)
  - `data/emnlp2026/wsb/benign_stress.jsonl`
  - `data/predictions/stander_jailbreak_eval_qwen/*.csv` (in-dist prompts)

Usage:
  python pivot_mistral_layer_profile.py --gpu 2
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PIV  = REPO / "data/emnlp2026/pivot"
WSB  = REPO / "data/emnlp2026/wsb"
PRED = REPO / "data/predictions"
HF   = "unsloth/mistral-7b-instruct-v0.3"
LAYERS = [4, 8, 12, 16, 20, 24, 28]
CHAT_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def uc(t):
    t = str(t); m = CHAT_RE.search(t)
    if m: return m.group(1).strip()
    for g in ("<|im_start|>", "<|im_end|>", "<|begin_of_text|>",
              "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"):
        t = t.replace(g, " ")
    return re.sub(r"\s+", " ", t).strip()


def extract_layers(gpu: int, batch_size: int = 16, max_length: int = 800):
    """GPU: forward-pass eval + bs once, cache hidden states at LAYERS."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import pandas as pd

    out_path = PIV / "mistral_hs_alllayers.npz"
    if out_path.exists():
        print(f"[skip] {out_path}"); return

    dev = torch.device(f"cuda:{gpu}")
    tok = AutoTokenizer.from_pretrained(HF)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        HF, torch_dtype=torch.bfloat16).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)

    # in-dist 4000 prompts
    df = pd.read_csv(sorted(glob.glob(str(PRED / "stander_jailbreak_eval_qwen"
                                          / "*_predictions.csv")))[0])
    prompts_in = [uc(t) for t in df["text"].tolist()]
    labels_in = (df["label"].values == 0).astype(int)        # jb=1 benign=0

    rows = [json.loads(l) for l in
            (WSB / "benign_stress.jsonl").read_text().splitlines() if l.strip()]
    prompts_bs = [r["prompt"] for r in rows]
    sources_bs = [r["dataset"] for r in rows]

    prompts = prompts_in + prompts_bs
    print(f"[extract] mistral L{LAYERS}, total {len(prompts)} prompts", flush=True)

    H = np.zeros((len(prompts), len(LAYERS), model.config.hidden_size),
                 dtype=np.float32)
    for i in range(0, len(prompts), batch_size):
        chats = [tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=False)
            for p in prompts[i:i + batch_size]]
        enc = tok(chats, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, add_special_tokens=True).to(dev)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        for j, L in enumerate(LAYERS):
            H[i:i + batch_size, j] = out.hidden_states[L][:, -1, :].float().cpu().numpy()
        if (i // batch_size) % 25 == 0:
            print(f"  [extract] {i + enc.input_ids.shape[0]}/{len(prompts)}",
                  flush=True)
        del out, enc

    np.savez_compressed(out_path,
                        H=H,
                        labels=np.array(labels_in, dtype=np.int64),
                        bs_ds=np.array(sources_bs),
                        n_in=len(prompts_in),
                        layers=np.array(LAYERS, dtype=np.int64))
    print(f"[extract] wrote {out_path}")


def analyze_layers():
    """CPU: per-layer probe AUC + LEACE source-id singular value spectrum."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    z = np.load(PIV / "mistral_hs_alllayers.npz", allow_pickle=True)
    H = z["H"]; labels = z["labels"]; bs_ds = z["bs_ds"]
    n_in = int(z["n_in"]); layers = z["layers"]
    Hin, Hbs = H[:n_in], H[n_in:]
    rng = np.random.default_rng(42)
    n_b = (labels == 0).sum(); n_j = (labels == 1).sum()
    ben = np.where(labels == 0)[0]; jb = np.where(labels == 1)[0]
    rng.shuffle(ben); rng.shuffle(jb)
    cal_b = ben[:n_b // 2]; te_b = ben[n_b // 2:]
    cal_j = jb[:n_j // 2]; te_j = jb[n_j // 2:]
    cal = np.array(sorted(list(cal_b) + list(cal_j)))
    tst = np.array(sorted(list(te_b) + list(te_j)))
    y = labels

    out = {"backbone": "mistral_aligned", "hf": HF,
           "layers": list(map(int, layers)),
           "per_layer": {}}
    for li, L in enumerate(layers):
        # probe AUC
        Xtr = Hin[cal, li]; ytr = y[cal]
        Xte = Hin[tst, li]; yte = y[tst]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
        p = clf.predict_proba(sc.transform(Xte))[:, 1]
        auc = float(roc_auc_score(yte, p))
        # LEACE source-id singular spectrum: M = isq @ X.T @ Z / n
        # use cal_benign + ALL bs as Xfit (matching heldout_remedy pattern but
        # all bs sources here since this is a profile, not a deployment).
        Xfit = np.vstack([Hin[cal_b, li], Hbs[:, li]])
        sources = sorted(set(map(str, bs_ds)))
        src = ["indist_benign"] + sources
        zlab = np.array([0] * len(cal_b) +
                        [src.index(str(s)) for s in bs_ds])
        Zoh = np.eye(len(src))[zlab][:, 1:]
        mu = Xfit.mean(0); Xc = Xfit - mu; eps = 1e-3
        Sig = (Xc.T @ Xc) / len(Xc) + eps * np.eye(Xc.shape[1])
        w, U = np.linalg.eigh(Sig); w = np.clip(w, eps, None)
        isq = (U * w ** -0.5) @ U.T
        Zc = Zoh - Zoh.mean(0)
        M = isq @ ((Xc.T @ Zc) / len(Xc))
        sm = np.linalg.svd(M, full_matrices=False, compute_uv=False)
        sm = sm.tolist()
        rank = int((np.array(sm) > 1e-6).sum())
        out["per_layer"][int(L)] = dict(
            probe_auc=round(auc, 5),
            leace_singular_values=[round(float(x), 5) for x in sm[:6]],
            leace_rank=rank)
        print(f"  L{int(L):2d}: AUC={auc:.4f}  rank={rank}  "
              f"sv={[round(x,3) for x in sm[:6]]}")
    op = PIV / "mistral_layer_profile.json"
    op.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {op}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--stage", choices=["extract", "analyze", "both"],
                    default="both")
    a = ap.parse_args()
    if a.stage in ("extract", "both"):
        extract_layers(a.gpu)
    if a.stage in ("analyze", "both"):
        analyze_layers()


if __name__ == "__main__":
    raise SystemExit(main())
