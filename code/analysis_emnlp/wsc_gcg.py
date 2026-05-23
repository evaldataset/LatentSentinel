#!/usr/bin/env python
"""WS-C (realistic): discrete GCG universal suffix vs the linear probe.

EMNLP 2026, Strategy Pillar 3. The embedding-space attack (wsc_adaptive_probe_
evasion.py) is the clean upper bound; this is the realistic case: a *discrete*
adversarial token suffix found by Greedy Coordinate Gradient (Zou et al. 2023),
universal (optimized on train jailbreaks, evaluated on held-out). White-box on the
published frozen backbone + linear head. Same recipe / conformal tau / split as the
embedding attack so the two are directly comparable.

Env: pinned in requirements.txt (see `bootstrap-latents` conda env)
  python wsc_gcg.py --model qwen_aligned --gpu 1
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
WSA = REPO / "data/emnlp2026/ws_a"
WSC = REPO / "data/emnlp2026/wsc"
PROBES = REPO / "data/trained_probes/stander_jailbreak"
PRED = REPO / "data/predictions"
MODELS = {
    "qwen_aligned": ("Qwen/Qwen2.5-7B-Instruct", "qwen", 3584, "{n}_lm_head.pth",
                     20, "stander_jailbreak_eval_qwen"),
    "llama_aligned": ("NousResearch/Meta-Llama-3.1-8B-Instruct", "llama", 4096,
                      "(Llama3.1)_{n}_lm_head.pth", 16, "stander_jailbreak_eval"),
}
CHAT_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def user_content(t):
    t = str(t)
    m = CHAT_RE.search(t)
    if m:
        return m.group(1).strip()
    for tg in ("<|im_start|>", "<|im_end|>", "<|begin_of_text|>",
               "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"):
        t = t.replace(tg, " ")
    return re.sub(r"\s+", " ", t).strip()


def jb_prompts(pred_dir, ntr, nte, seed):
    import pandas as pd
    f = sorted(glob.glob(str(PRED / pred_dir / "*_predictions.csv")))[0]
    df = pd.read_csv(f)
    t = [user_content(x) for x in df[df["label"] == 0]["text"]]
    t = [x for x in t if x and len(x) > 8]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(t))
    return [t[i] for i in idx[:ntr]], [t[i] for i in idx[ntr:ntr + nte]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--k", type=int, default=16, help="suffix length (tokens)")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--n-cand", type=int, default=64)
    ap.add_argument("--train-batch", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-test", type=int, default=512)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    hf, arch, hidden, ptmpl, layer, pred_dir = MODELS[a.model]
    probe_fp = PROBES / ptmpl.format(n=layer)
    b = json.loads((WSA / f"{pred_dir}__balanced.json").read_text())[
        "best_layer_by_auc"]
    tau = float(b["conformal"]["alpha_0.05"]["tau"])
    tr, te = jb_prompts(pred_dir, a.n_train, a.n_test, a.seed)
    print(f"[gcg/{a.model}] L{layer} tau={tau:.4f} train={len(tr)} test={len(te)} "
          f"k={a.k} steps={a.steps}", flush=True)

    import torch, torch.nn as nn
    from transformers import AutoTokenizer, Qwen2ForCausalLM, LlamaForCausalLM
    torch.manual_seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}")
    tok = AutoTokenizer.from_pretrained(hf)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    LM = Qwen2ForCausalLM if arch == "qwen" else LlamaForCausalLM
    model = LM.from_pretrained(hf, torch_dtype=torch.float32)
    model.model.layers = nn.ModuleList(model.model.layers[:layer])
    model.lm_head = nn.Linear(hidden, 2, bias=False)
    sd = torch.load(probe_fp, map_location="cpu")
    model.lm_head.load_state_dict(sd if "weight" in sd else {"weight": sd["weight"]})
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    emb = model.get_input_embeddings()
    W = emb.weight                                   # (V, d)
    V = W.shape[0]

    def tok_prompt(p):
        # CHECK fix C-3/H-4: unified with wsc_adaptive_probe_evasion.py.
        # - add_generation_prompt=False (probe trained without it)
        # - add_special_tokens=True (matches probe training)
        s = tok.apply_chat_template([{"role": "user", "content": p}],
                                    tokenize=False,
                                    add_generation_prompt=False)
        return tok(s, truncation=True, max_length=a.max_length,
                   add_special_tokens=True, return_tensors="pt").input_ids[0]

    tr_ids = [tok_prompt(p).to(dev) for p in tr]
    rng = np.random.default_rng(a.seed)
    # init suffix = "! ! ! ..." style (a single benign-ish token id repeated)
    init_id = tok(" !", add_special_tokens=False).input_ids[-1]
    suffix = torch.full((a.k,), init_id, device=dev, dtype=torch.long)

    def pjb_batch(prompt_ids_list, suf):
        """mean P(jailbreak) over a list of prompts with suffix `suf` appended."""
        outs = []
        for pid in prompt_ids_list:
            ids = torch.cat([pid, suf])
            e = emb(ids).unsqueeze(0)
            lg = model(inputs_embeds=e).logits[:, -1, :].float()
            outs.append(torch.softmax(lg, -1)[0, 0])     # class0 = jailbreak
        return torch.stack(outs)

    for step in range(a.steps):
        bi = rng.choice(len(tr_ids), min(a.train_batch, len(tr_ids)),
                        replace=False)
        batch = [tr_ids[i] for i in bi]
        # one-hot suffix with grad -> gradient wrt token choice
        oh = torch.zeros(a.k, V, device=dev, requires_grad=True)
        with torch.no_grad():
            oh.scatter_(1, suffix.unsqueeze(1), 1.0)
        oh.requires_grad_(True)
        suf_emb = oh @ W
        loss = 0.0
        for pid in batch:
            e = torch.cat([emb(pid), suf_emb], 0).unsqueeze(0)
            lg = model(inputs_embeds=e).logits[:, -1, :].float()
            loss = loss + torch.softmax(lg, -1)[0, 0]
        loss = loss / len(batch)
        grad = torch.autograd.grad(loss, oh)[0]          # (k, V)
        # candidate suffixes: replace a random position with a top-k token
        with torch.no_grad():
            topk = (-grad).topk(a.topk, dim=1).indices   # (k, topk)
            cands = suffix.repeat(a.n_cand, 1)
            pos = torch.randint(0, a.k, (a.n_cand,), device=dev)
            pick = topk[pos, torch.randint(0, a.topk, (a.n_cand,), device=dev)]
            cands[torch.arange(a.n_cand), pos] = pick
            best, best_l = suffix, loss.item()
            for c in cands:
                v = pjb_batch(batch, c).mean().item()
                if v < best_l:
                    best_l, best = v, c.clone()
            suffix = best
        if step % 20 == 0 or step == a.steps - 1:
            print(f"  [gcg] step {step:3d} mean P(jb|train)={best_l:.4f}",
                  flush=True)

    @torch.no_grad()
    def recall(prompts, suf):
        flagged = 0
        for p in prompts:
            pid = tok_prompt(p).to(dev)
            ids = torch.cat([pid, suf]) if suf is not None else pid
            lg = model(inputs_embeds=emb(ids).unsqueeze(0)).logits[:, -1, :].float()
            if torch.softmax(lg, -1)[0, 0].item() >= tau:
                flagged += 1
        return flagged / len(prompts)

    r_before = recall(te, None)
    r_after = recall(te, suffix)
    suf_txt = tok.decode(suffix)
    res = {"model": a.model, "layer": layer, "tau": round(tau, 4),
           "k": a.k, "steps": a.steps, "n_test": len(te),
           "recall_before": round(r_before, 4),
           "recall_after": round(r_after, 4),
           "evasion_rate": round((r_before - r_after) / max(r_before, 1e-9), 4),
           "suffix_text": suf_txt}
    WSC.mkdir(parents=True, exist_ok=True)
    out = WSC / "gcg_evasion.json"
    allr = json.loads(out.read_text()) if out.exists() else {}
    allr[a.model] = res
    out.write_text(json.dumps(allr, indent=1))
    print(f"\n== GCG (discrete) / {a.model} ==")
    print(f"   recall@τ {r_before:.3f} -> {r_after:.3f}  "
          f"evasion {res['evasion_rate']:.3f}")
    print(f"   suffix: {suf_txt!r}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
