#!/usr/bin/env python
"""WS-B (GPU): over-refusal of Latent Sentinel on diverse benign traffic.

EMNLP 2026, Strategy Pillar 2. Runs the cached benign-stress prompts
(data/emnlp2026/wsb/benign_stress.jsonl) through a backbone truncated to the WS-A
best layer + the saved linear probe, EXACTLY replicating the training/eval recipe in
code/probe_training/jailbreak_layer_15.py:

  AutoTokenizer.apply_chat_template([{user: prompt}], tokenize=False)
  model.model.layers = model.model.layers[:N]
  model.lm_head = nn.Linear(hidden, 2, bias=False)  <- load (N)_lm_head.pth
  logits = model(...).logits[:, -1, :]              # last token
  P(jailbreak) = softmax(logits)[:, 0]              # class 0 == jailbreak

Then applies the WS-A thresholds (default 0.5, balanced-cal, and the conformal tau
calibrated on IN-DISTRIBUTION Alpaca-style benign) and reports per-dataset
false-positive rate. Hypothesis/finding: a conformal benign-FPR guarantee calibrated
on narrow benign does NOT hold on diverse benign -> over-refusal of real users.

GPU-blocked on shared machines: use --wait-for-vram to poll until free.
Env: pinned in requirements.txt (see `bootstrap-latents` conda env)
  python wsb_benign_stress_infer.py --model qwen_aligned --wait-for-vram 16
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WSB = REPO / "data/emnlp2026/wsb"
WSA = REPO / "data/emnlp2026/ws_a"
PROBES = REPO / "data/trained_probes/stander_jailbreak"

# model key -> (HF id or local path, arch, hidden, probe-file template, WS-A best layer,
#               WS-A source json stem for conformal tau)
MODELS = {
    "qwen_aligned": ("Qwen/Qwen2.5-7B-Instruct", "qwen", 3584,
                     "{n}_lm_head.pth", 20, "stander_jailbreak_eval_qwen"),
    # meta-llama is gated for our token; NousResearch mirror is ungated & weight-identical
    "llama_aligned": ("NousResearch/Meta-Llama-3.1-8B-Instruct", "llama", 4096,
                      "(Llama3.1)_{n}_lm_head.pth", 16, "stander_jailbreak_eval"),
    "qwen_jb": (str(REPO / "data/alignment_degraded/Qwen_2.5_jailbreak_extracted"),
                "qwen", 3584, "{n}_lm_head.pth", 16,
                "stander_jailbreak_eval_qwen_jailbreak"),
    "llama_jb": (str(REPO / "data/alignment_degraded/Llama_3.1_jailbreak_weights"),
                 "llama", 4096, "(Llama3.1)_{n}_lm_head.pth", 15,
                 "stander_jailbreak_eval_jailbreak"),
}


def free_vram_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"], text=True)
    return {int(i): int(f) for i, f in
            (ln.split(", ") for ln in out.strip().splitlines())}


def pick_gpu(need_gib: float, wait: bool, force=None):
    need = int(need_gib * 1024)
    if force is not None:
        fv = free_vram_mib()
        print(f"[gpu] pinned cuda:{force} ({fv.get(force, '?')} MiB free, need {need})")
        return force
    while True:
        fv = free_vram_mib()
        g, mib = max(fv.items(), key=lambda kv: kv[1])
        if mib >= need:
            print(f"[gpu] using cuda:{g} ({mib} MiB free, need {need})")
            return g
        if not wait:
            print(f"[gpu] BLOCKED: max free {mib} MiB < {need} MiB needed. "
                  f"Re-run with --wait-for-vram or free a GPU.", file=sys.stderr)
            sys.exit(3)
        print(f"[gpu] waiting: max free {mib} MiB < {need} MiB ... (sleep 120s)")
        time.sleep(120)


def conformal_taus(stem: str):
    p = WSA / f"{stem}__balanced.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    b = d["best_layer_by_auc"]
    out = {"default_0p5": 0.5,
           "balanced_cal": b["op_balanced_cal"]["threshold"]}
    # WS-A wrote conformal keys via f"alpha_{a}" -> 'alpha_0.01','alpha_0.05','alpha_0.1'.
    # Pass them through verbatim (prefixed) so this never desyncs from WS-A's format.
    for k, v in b.get("conformal", {}).items():
        out[f"conformal_{k}"] = v["tau"]   # -> conformal_alpha_0.01 / _0.05 / _0.1
    return out


def run_one(mkey, a, dev, prompts, dsets):
    import torch, torch.nn as nn, numpy as np
    from transformers import AutoTokenizer, Qwen2ForCausalLM, LlamaForCausalLM
    hf_or_path, arch, hidden, ptmpl, best, stem = MODELS[mkey]
    layer = a.layer or best
    probe_fp = PROBES / ptmpl.format(n=layer)
    if not probe_fp.exists():
        print(f"[{mkey}] SKIP — probe weights missing: {probe_fp}", file=sys.stderr)
        return None
    is_hf = ("/" in hf_or_path) and not Path(hf_or_path).exists()
    if not (is_hf or Path(hf_or_path).exists()):
        print(f"[{mkey}] SKIP — backbone path missing: {hf_or_path}", file=sys.stderr)
        return None
    taus = conformal_taus(stem)   # fail fast on threshold-lookup bugs, before GPU load
    if not taus:
        print(f"[{mkey}] SKIP — no WS-A thresholds for stem {stem}", file=sys.stderr)
        return None
    print(f"[{mkey}] loading {hf_or_path} (L{layer}, probe {probe_fp.name}) ...")
    # The alignment-degraded dirs are weights-only (no tokenizer files); SFT does
    # not change the tokenizer, so fall back to the base model's tokenizer by arch
    # (this is what the original training script did).
    tok_base = ("Qwen/Qwen2.5-7B-Instruct" if arch == "qwen"
                else "NousResearch/Meta-Llama-3.1-8B-Instruct")
    try:
        tok = AutoTokenizer.from_pretrained(hf_or_path)
    except Exception as e:  # noqa: BLE001
        print(f"[{mkey}] tokenizer absent in backbone ({type(e).__name__}); "
              f"using base {tok_base}")
        tok = AutoTokenizer.from_pretrained(tok_base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Left-pad so logits[:, -1, :] is always the last *real* token (the method's
    # "last token" readout), independent of batch padding -> lets us use dynamic
    # (longest-in-batch) padding instead of fixed max_length=800: ~10-200x faster
    # with identical last-token logits under causal+masked attention.
    tok.padding_side = "left"
    # bf16 backbone (A100-fast). The probe is a 2-class linear readout; bf16 is
    # ample for an over-refusal/FPR *characterization* (gross effects, coarse tau).
    # --dtype fp32 available for an exact WS-A-reproduction check on a subset.
    dt = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    kw = dict(torch_dtype=dt)
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw = dict(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))
    LM = Qwen2ForCausalLM if arch == "qwen" else LlamaForCausalLM
    model = LM.from_pretrained(hf_or_path, **kw)
    model.model.layers = nn.ModuleList(model.model.layers[:layer])
    model.lm_head = nn.Linear(hidden, 2, bias=False)
    sd = torch.load(probe_fp, map_location="cpu")
    model.lm_head.load_state_dict(sd if "weight" in sd else {"weight": sd["weight"]})
    model.lm_head = model.lm_head.to(dtype=next(model.model.parameters()).dtype)
    if not a.load_4bit:
        model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Robust chat formatting: some saved tokenizers (e.g. the alignment-degraded
    # Llama dir) have no chat_template -> apply_chat_template errors / returns
    # non-str. Coerce to str, try the template, else fall back to the Llama-3 / a
    # generic user template (matches what training's apply_chat_template emits).
    has_tmpl = getattr(tok, "chat_template", None) is not None

    def fmt(c):
        c = "" if c is None else str(c)
        if has_tmpl:
            try:
                t = tok.apply_chat_template([{"role": "user", "content": c}],
                                            tokenize=False)
                if isinstance(t, str):
                    return t
            except Exception:  # noqa: BLE001
                pass
        return ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{c}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

    scores = []
    with torch.no_grad():
        for i in range(0, len(prompts), a.batch_size):
            chunk = prompts[i:i + a.batch_size]
            texts = [fmt(c) for c in chunk]
            enc = tok(texts, truncation=True, padding=True,
                      max_length=a.max_length, return_tensors="pt").to(dev)
            lg = model(**enc).logits[:, -1, :].float()  # left-pad -> last real tok
            scores.extend(torch.softmax(lg, -1)[:, 0].cpu().tolist())  # class0=jailbreak
            if (i // a.batch_size) % 25 == 0:
                print(f"  [{mkey}] {i+len(chunk)}/{len(prompts)}", flush=True)
    s = np.array(scores)
    np.save(WSB / f"scores_{mkey}.npy", s)   # persist GPU work before any post-proc
    res = {"model": mkey, "backbone": hf_or_path, "layer": layer,
           "probe": str(probe_fp), "n": len(s), "thresholds": taus, "per_dataset": {}}
    for ds in sorted(set(dsets)):
        m = np.array([d == ds for d in dsets])
        sd_ = s[m]
        res["per_dataset"][ds] = {
            "n": int(m.sum()), "mean_jb_score": round(float(sd_.mean()), 4),
            "FPR": {k: round(float((sd_ >= t).mean()), 4) for k, t in taus.items()}}
    res["overall_FPR"] = {k: round(float((s >= t).mean()), 4) for k, t in taus.items()}
    del model
    torch.cuda.empty_cache()
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--models", help="comma list or 'all'")
    ap.add_argument("--gpu", type=int, help="pin this GPU index (skip auto-pick)")
    ap.add_argument("--layer", type=int, help="override WS-A best layer")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                    help="bf16 (fast, default) | fp32 (exact WS-A repro check)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=800)
    ap.add_argument("--load-4bit", action="store_true",
                    help="bitsandbytes 4-bit (fits ~6GB; risky under contention)")
    ap.add_argument("--wait-for-vram", type=float, default=0.0,
                    help="GiB to wait for (poll every 120s) instead of failing")
    ap.add_argument("--out", type=Path, default=WSB / "benign_stress_fpr.json")
    a = ap.parse_args()
    if a.models == "all":
        mkeys = list(MODELS)
    elif a.models:
        mkeys = [m.strip() for m in a.models.split(",")]
    elif a.model:
        mkeys = [a.model]
    else:
        ap.error("give --model X or --models a,b,.. or --models all")

    src = WSB / "benign_stress.jsonl"
    if not src.exists():
        print("run wsb_fetch_benign_stress.py first", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    prompts = [r["prompt"] for r in rows]
    dsets = [r["dataset"] for r in rows]

    import torch
    need = 6.0 if a.load_4bit else 17.0
    gpu = pick_gpu(a.wait_for_vram or need, wait=a.wait_for_vram > 0, force=a.gpu)
    dev = torch.device(f"cuda:{gpu}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    allres = json.loads(a.out.read_text()) if a.out.exists() else {}
    for mkey in mkeys:
        try:
            r = run_one(mkey, a, dev, prompts, dsets)
        except Exception as e:  # noqa: BLE001 - keep batch going
            print(f"[{mkey}] ERROR {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if r is None:
            continue
        allres[mkey] = r
        a.out.write_text(json.dumps(allres, indent=2))  # checkpoint each model
        print(f"\n== {mkey}  L{r['layer']}  (n={r['n']} diverse-benign prompts; "
              f"conformal τ calibrated for FPR≤α on IN-DIST benign)")
        for ds, v in r["per_dataset"].items():
            print(f"   {ds:20s} n={v['n']:4d}  meanJB={v['mean_jb_score']}  "
                  f"FPR@0.5={v['FPR'].get('default_0p5')}  "
                  f"FPR@conf.05={v['FPR'].get('conformal_alpha_0.05')}  "
                  f"FPR@conf.10={v['FPR'].get('conformal_alpha_0.1')}")
        print(f"   OVERALL FPR @conf.05 = {r['overall_FPR'].get('conformal_alpha_0.05')} "
              f"(guarantee was ≤0.05 on in-dist Alpaca-style benign)")
    print(f"\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
