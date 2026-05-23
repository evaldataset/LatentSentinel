#!/usr/bin/env python
"""WS-D: training-free representation baseline (HiddenDetect-style) vs our probe.

EMNLP 2026, Strategy §3/WS-D. The #1 reviewer question for a trained-probe paper:
"does training buy anything over a zero-training representation readout, under honest
FPR-controlled metrics?" We implement a HiddenDetect-style detector (Jiang et al.,
ACL 2025, arXiv:2502.14744): with the model's ORIGINAL unembedding, project each
layer's last-token hidden state to vocab, measure refusal-token probability mass
(the model's internal "I should refuse" signal), aggregate over mid/late layers.
NO training. Evaluate on the SAME data as our probe:
  (a) in-distribution balanced eval: 2000 benign + 2000 jailbreak from
      data/predictions/<dir>/ (text+label), and
  (b) the WS-B diverse-benign stress set (data/emnlp2026/wsb/benign_stress.jsonl),
report ROC-AUC, and FPR at a conformal tau calibrated on the SAME held-out in-dist
benign (identical protocol to WS-A) — so the comparison is apples-to-apples.

Only the 2 recoverable aligned backbones (both *_JB artifacts are incomplete).
Env: pinned in requirements.txt (see `bootstrap-latents` conda env)
  python wsd_hiddendetect_baseline.py --model qwen_aligned --gpu 2
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WSB = REPO / "data/emnlp2026/wsb"
WSD = REPO / "data/emnlp2026/wsd"
PRED = REPO / "data/predictions"

MODELS = {
    "qwen_aligned": ("Qwen/Qwen2.5-7B-Instruct", "stander_jailbreak_eval_qwen"),
    "llama_aligned": ("NousResearch/Meta-Llama-3.1-8B-Instruct",
                      "stander_jailbreak_eval"),
    "mistral_aligned": ("unsloth/mistral-7b-instruct-v0.3",
                        "stander_jailbreak_eval_qwen"),
}
# Refusal-onset vocabulary (HiddenDetect uses the model's tendency to emit refusals).
REFUSAL_WORDS = ["I", "Sorry", "sorry", "cannot", "can't", "unable", "apolog",
                 "unfortunately", "Unfortunately", "illegal", "not able", "won't",
                 "will not", "I'm sorry", "As an", "I cannot", "refuse"]
CHAT_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def user_content(t: str) -> str:
    t = str(t)
    m = CHAT_RE.search(t)
    if m:
        return m.group(1).strip()
    for tg in ("<|im_start|>", "<|im_end|>", "<|begin_of_text|>",
               "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"):
        t = t.replace(tg, " ")
    return re.sub(r"\s+", " ", t).strip()


def load_indist(pred_dir: str):
    f = sorted(glob.glob(str(PRED / pred_dir / "*_predictions.csv")))[0]
    import pandas as pd
    df = pd.read_csv(f)
    txt = [user_content(t) for t in df["text"].tolist()]
    y = (df["label"].values == 0).astype(int)        # 1 == jailbreak (positive)
    return txt, y


def conformal_tau_from(scores_benign_cal: np.ndarray, alpha=0.05) -> float:
    s = np.sort(scores_benign_cal)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[min(k, n) - 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    hf, pred_dir = MODELS[a.model]

    txt, y = load_indist(pred_dir)
    bstress = [json.loads(l) for l in
               (WSB / "benign_stress.jsonl").read_text().splitlines() if l.strip()]
    bs_txt = [r["prompt"] for r in bstress]
    bs_ds = [r["dataset"] for r in bstress]
    print(f"[{a.model}] in-dist n={len(txt)} (jb={int(y.sum())}/ben={int((1-y).sum())})"
          f"  benign-stress n={len(bs_txt)}")

    import torch, torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}")
    tok = AutoTokenizer.from_pretrained(hf)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False

    # refusal token id set (first sub-token of each cue, both raw and space-prefixed)
    rid = set()
    for w in REFUSAL_WORDS:
        for v in (w, " " + w):
            ids = tok(v, add_special_tokens=False)["input_ids"]
            if ids:
                rid.add(ids[0])
    rid = torch.tensor(sorted(rid), device=dev)
    nlayers = model.config.num_hidden_layers
    band = list(range(nlayers // 2, int(nlayers * 0.9) + 1))  # safety-aware band

    @torch.no_grad()
    def hd_scores(prompts):
        out = []
        for i in range(0, len(prompts), a.batch_size):
            chunk = [tok.apply_chat_template([{"role": "user", "content": c}],
                                             tokenize=False) for c in prompts[i:i + a.batch_size]]
            enc = tok(chunk, truncation=True, padding=True, max_length=a.max_length,
                      return_tensors="pt").to(dev)
            hs = model(**enc, output_hidden_states=True).hidden_states  # tuple L+1
            W = model.get_output_embeddings().weight                    # (V,H)
            s = torch.zeros(enc["input_ids"].shape[0], device=dev)
            for L in band:
                h = hs[L][:, -1, :]                                     # last (left-pad)
                logits = h.float() @ W.float().T
                p = torch.softmax(logits, -1)
                s += p[:, rid].sum(-1)                                  # refusal mass
            out.extend((s / len(band)).cpu().tolist())
            if (i // a.batch_size) % 30 == 0:
                print(f"  [{a.model}] {i+len(chunk)}/{len(prompts)}", flush=True)
        return np.array(out)

    s_ind = hd_scores(txt)               # higher => more "refusal" => predict jailbreak
    s_bs = hd_scores(bs_txt)

    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, s_ind))
    # identical WS-A protocol: split in-dist benign -> calibrate conformal tau on half
    rng = np.random.default_rng(a.seed)
    ben_idx = np.where(y == 0)[0]
    rng.shuffle(ben_idx)
    cut = len(ben_idx) // 2
    tau = conformal_tau_from(s_ind[ben_idx[:cut]], 0.05)
    test_ben = ben_idx[cut:]
    jb_idx = np.where(y == 1)[0]
    res = {
        "model": a.model, "method": "HiddenDetect-style (training-free)",
        "refusal_band_layers": [band[0], band[-1]], "n_refusal_tokens": int(len(rid)),
        "roc_auc_indist": round(auc, 4),
        "tau_conformal_a0.05_on_indist_benign": round(tau, 4),
        "indist_jailbreak_recall_at_tau": round(float((s_ind[jb_idx] >= tau).mean()), 4),
        "indist_benign_fpr_at_tau": round(float((s_ind[test_ben] >= tau).mean()), 4),
        "benign_stress_FPR_at_tau": {},
    }
    for ds in sorted(set(bs_ds)):
        m = np.array([d == ds for d in bs_ds])
        res["benign_stress_FPR_at_tau"][ds] = round(float((s_bs[m] >= tau).mean()), 4)
    res["benign_stress_FPR_overall"] = round(float((s_bs >= tau).mean()), 4)

    WSD.mkdir(parents=True, exist_ok=True)
    out = WSD / "hiddendetect_baseline.json"
    allr = json.loads(out.read_text()) if out.exists() else {}
    allr[a.model] = res
    out.write_text(json.dumps(allr, indent=2))
    np.savez(WSD / f"scores_{a.model}.npz", indist=s_ind, y=y, bstress=s_bs)
    print(f"\n== {a.model}  HiddenDetect-style (training-free) ==")
    print(f"   in-dist ROC-AUC = {res['roc_auc_indist']} "
          f"(our trained probe WS-A best-layer AUC ~0.96-1.00)")
    print(f"   @conformal τ: in-dist jb recall={res['indist_jailbreak_recall_at_tau']}"
          f"  in-dist benign FPR={res['indist_benign_fpr_at_tau']}")
    print(f"   benign-stress overall FPR @τ = {res['benign_stress_FPR_overall']} "
          f"(our probe: Qwen 0.982 / Llama 0.683)")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
