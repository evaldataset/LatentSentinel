#!/usr/bin/env python
"""PIVOT spine + remedy: benign-REGISTER erasure and its OOD effect.

==============================================================================
CHECK.md AUDIT FIX (C-2): The remedy JSON this script *originally* produced has
transductive LEACE leakage (eraser fit on ALL Hbs, then evaluated on the same).
The leakage-free spine is `pivot_heldout_remedy.py` (3-way disjoint split).

THIS SCRIPT IS RETAINED AS THE UNIQUE PRODUCER OF THE `hs_*.npz` CACHES.
The cached hidden-state files (`hs_qwen_aligned_L20.npz`,
`hs_llama_aligned_L16.npz`, `hs_mistral_aligned_L16.npz`) are required inputs
for every downstream `pivot_*` analysis. The remedy JSON it ALSO emits is
written as `register_erase_remedy.SUPERSEDED.json` so no paper table can cite
it by accident. Use `heldout_remedy.json` for all paper numbers.
==============================================================================

The decisive experiment the cached run could not do (cache has no diverse benign).
Phase E (GPU): extract last-token hidden states at the WS-A best layer for the
in-distribution eval (4000: 2000 benign Alpaca-style + 2000 jailbreak) and the
benign-stress set (3567: XSTest/OR-Bench/ToxicChat). Phase A (no GPU): fit a
closed-form LEACE eraser for the BENIGN-REGISTER concept Z = benign-source identity
(Alpaca-style vs each diverse-benign source), using BENIGN samples only so the
jailbreak label Y never enters Z (amnesic). Then compare, before vs after erasure:
  (i) in-dist test ROC-AUC (jailbreak vs benign) and jailbreak recall,
  (ii) benign-stress FPR at a conformal tau calibrated on in-dist benign.

Env: pinned in `requirements.txt` (or use the bootstrap-latents conda env).
  python pivot_register_erase_remedy.py --model qwen_aligned --gpu auto --wait-vram 17
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, subprocess, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
WSB, PIV = REPO / "data/emnlp2026/wsb", REPO / "data/emnlp2026/pivot"
PRED = REPO / "data/predictions"
MODELS = {"qwen_aligned": ("Qwen/Qwen2.5-7B-Instruct", "stander_jailbreak_eval_qwen", 20),
          "llama_aligned": ("NousResearch/Meta-Llama-3.1-8B-Instruct",
                            "stander_jailbreak_eval", 16),
          # Third, architecturally-distinct backbone (different family/tokenizer)
          # for cross-model generalization. CHECK fix B4: switched to ungated
          # mirror so the held-out third backbone is reachable without HF gating.
          # Layer 16 is fixed a priori (mid-network, NOT WS-A-tuned).
          "mistral_aligned": ("unsloth/mistral-7b-instruct-v0.3",
                              "stander_jailbreak_eval_qwen", 16)}
CHAT_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def uc(t):
    t = str(t); m = CHAT_RE.search(t)
    if m: return m.group(1).strip()
    for g in ("<|im_start|>", "<|im_end|>", "<|begin_of_text|>",
              "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"):
        t = t.replace(g, " ")
    return re.sub(r"\s+", " ", t).strip()


def free_gpu(need_mib):
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                   "--format=csv,noheader,nounits"], text=True)
    fv = {int(i): int(f) for i, f in (l.split(", ") for l in out.strip().splitlines())}
    g, f = max(fv.items(), key=lambda kv: kv[1])
    return (g if f >= need_mib else None), f


# ---------- closed-form LEACE (fit on a chosen subset) ----------
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


def cv_auc(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                          X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          method="predict_proba", n_jobs=4)[:, 1]
    return float(roc_auc_score(y, p)), p


def fit_eval(Htr, ytr, Hte, yte, Hbs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Htr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Htr), ytr)
    s_te = clf.predict_proba(sc.transform(Hte))[:, 1]
    s_bs = clf.predict_proba(sc.transform(Hbs))[:, 1]
    auc = float(roc_auc_score(yte, s_te))
    ben = s_te[yte == 0]
    tau = float(np.sort(ben)[int(np.ceil((len(ben) + 1) * 0.95)) - 1])  # conformal a=.05
    jb_recall = float((s_te[yte == 1] >= tau).mean())
    bs_fpr = float((s_bs >= tau).mean())
    return dict(auc=round(auc, 4), tau=round(tau, 4),
                jb_recall_at_tau=round(jb_recall, 4),
                benign_stress_FPR_at_tau=round(bs_fpr, 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--wait-vram", type=int, default=0, help="MiB to poll-wait for")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    a = ap.parse_args()
    hf, pdir, layer = MODELS[a.model]
    PIV.mkdir(parents=True, exist_ok=True)
    cache = PIV / f"hs_{a.model}_L{layer}.npz"

    if not cache.exists():
        need = 17000
        if a.wait_vram:
            while True:
                g, f = free_gpu(a.wait_vram)
                if g is not None: break
                print(f"[wait] max free {f}MiB < {a.wait_vram}; sleep 120", flush=True)
                time.sleep(120)
        else:
            g, f = free_gpu(need)
            if g is None:
                print(f"[BLOCKED] max free {f}MiB < {need}; rerun with --wait-vram",
                      file=sys.stderr); return 3
        gpu = g if a.gpu == "auto" else int(a.gpu)
        import pandas as pd, torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = torch.device(f"cuda:{gpu}")
        df = pd.read_csv(sorted(glob.glob(str(PRED / pdir / "*_predictions.csv")))[0])
        in_txt = [uc(t) for t in df["text"]]
        y = (df["label"].values == 0).astype(int)              # jailbreak=1
        rows = [json.loads(l) for l in
                (WSB / "benign_stress.jsonl").read_text().splitlines() if l.strip()]
        bs_txt = [r["prompt"] for r in rows]; bs_ds = [r["dataset"] for r in rows]
        tok = AutoTokenizer.from_pretrained(hf)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            hf, torch_dtype=torch.bfloat16).to(dev).eval()
        for p in model.parameters(): p.requires_grad_(False)

        def emb(prompts):
            o = []
            for i in range(0, len(prompts), a.batch_size):
                ms = [[{"role": "user", "content": p}]
                      for p in prompts[i:i + a.batch_size]]
                enc = tok.apply_chat_template(
                    ms, add_generation_prompt=True, return_tensors="pt",
                    padding=True, truncation=True, max_length=a.max_length,
                    return_dict=True).to(dev)
                with torch.no_grad():
                    h = model(**enc, output_hidden_states=True
                              ).hidden_states[layer][:, -1, :].float().cpu().numpy()
                o.append(h)
                if (i // a.batch_size) % 25 == 0:
                    print(f"  emb {i+len(ms)}/{len(prompts)}", flush=True)
            return np.concatenate(o)
        Hin = emb(in_txt); Hbs = emb(bs_txt)
        np.savez(cache, Hin=Hin, y=y, Hbs=Hbs,
                 bs_ds=np.array(bs_ds), layer=layer)
        print(f"[extract] cached {cache}", flush=True)

    z = np.load(cache, allow_pickle=True)
    Hin, y, Hbs, bs_ds = z["Hin"], z["y"], z["Hbs"], z["bs_ds"]
    rng = np.random.default_rng(42)
    ben = np.where(y == 0)[0]; jb = np.where(y == 1)[0]
    rng.shuffle(ben); rng.shuffle(jb)
    cal_b, te_b = ben[:len(ben)//2], ben[len(ben)//2:]
    tr = np.array(sorted(list(cal_b) + list(jb[:len(jb)//2])))
    te = np.array(sorted(list(te_b) + list(jb[len(jb)//2:])))

    # Benign-register concept Z: in-dist benign = source 0; benign-stress sources 1..k.
    # Fit eraser on BENIGN ONLY (in-dist benign + benign-stress) -> Y never in Z.
    src_names = ["indist_benign"] + sorted(set(map(str, bs_ds)))
    Xfit = np.vstack([Hin[ben], Hbs])
    zlab = np.array([0] * len(ben) + [src_names.index(str(s)) for s in bs_ds])
    Zoh = np.eye(len(src_names))[zlab][:, 1:]
    mu, R, rnk = leace(Xfit, Zoh)
    er = lambda X: ((X - mu) @ R.T + mu).astype(np.float32)

    base = fit_eval(Hin[tr], y[tr], Hin[te], y[te], Hbs)
    ersd = fit_eval(er(Hin[tr]), y[tr], er(Hin[te]), y[te], er(Hbs))
    # Pivot #2: register-balanced training (mix half the benign-stress in as benign)
    nbs = len(Hbs) // 2
    Htr2 = np.vstack([Hin[tr], Hbs[:nbs]]); ytr2 = np.concatenate([y[tr], np.zeros(nbs, int)])
    rbt = fit_eval(Htr2, ytr2, Hin[te], y[te], Hbs[nbs:])
    # register recoverability sanity
    zr_b, _ = cv_auc(Xfit, (zlab > 0).astype(int))
    zr_a, _ = cv_auc(er(Xfit), (zlab > 0).astype(int))

    res = {"model": a.model, "layer": int(z["layer"]), "leace_rank": int(rnk),
           "baseline": base, "register_erased": ersd,
           "register_balanced_training": rbt,
           "Zrecover_benignsource_before": round(zr_b, 3),
           "Zrecover_benignsource_after": round(zr_a, 3)}
    out = PIV / "register_erase_remedy.SUPERSEDED.json"
    allr = json.loads(out.read_text()) if out.exists() else {}
    allr[a.model] = res
    out.write_text(json.dumps(allr, indent=1))
    print(f"\n== {a.model} L{res['layer']} (LEACE rank {rnk}) ==")
    for k in ("baseline", "register_erased", "register_balanced_training"):
        v = res[k]
        print(f"  {k:28s} AUC={v['auc']}  jb_recall@tau={v['jb_recall_at_tau']}  "
              f"benign-stress FPR@tau={v['benign_stress_FPR_at_tau']}")
    print(f"  benign-source recoverability {zr_b:.2f} -> {zr_a:.2f} (erasure effect)")
    d = base["benign_stress_FPR_at_tau"] - ersd["benign_stress_FPR_at_tau"]
    keep = ersd["auc"] >= base["auc"] - 0.03
    print("VERDICT:", "OUTCOME 1 (validated remedy: erasure keeps AUC & cuts OOD FPR "
          f"by {d:+.3f})" if (keep and d > 0.15) else
          "OUTCOME 2 (erasure destroys detection or doesn't fix OOD -> the linear "
          "signal IS benign-register; causal claim)")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
