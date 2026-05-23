#!/usr/bin/env python
"""B1 — RECIPE-AWARE GCG: discrete universal suffix optimized directly against
the RECIPE (LEACE-erased probe + deployment conformal tau), not the baseline.

The recipe's score reduces to a single effective linear head on the raw hidden
state:
  er(h)     = (h-mu) @ R.T + mu
  s(h)      = sigmoid( (coef/scale) . er(h)  +  intercept - (coef/scale).mean_ )
            = sigmoid( w_eff . h + b_eff )
with
  v       = coef / scale_
  w_eff   = R.T @ v
  b_eff   = (v - w_eff) . mu  + (intercept - v . mean_)
so GCG runs identically to wsc_gcg.py except the published probe lm_head is
replaced by this derived (w_eff, b_eff) recipe head.

For each backbone {qwen,llama}_aligned:
  (1) reconstruct the pivot_heldout_remedy split + eraser + scaled LogReg.
  (2) derive w_eff, b_eff; install as lm_head.
  (3) run universal-suffix GCG against held-in jailbreaks; eval on held-out.
  (4) compare recall@deploy_tau before/after, report evasion vs baseline-only
      GCG (from wsc/gcg_evasion.json).

Env: <conda-env>/bin/python
  python pivot_adaptive_gcg_recipe.py --model qwen_aligned --gpu 1
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PIV = REPO / "data/emnlp2026/pivot"
WSC = REPO / "data/emnlp2026/wsc"
PRED = REPO / "data/predictions"
ALPHA = 0.05
MODELS = {
    "qwen_aligned":  ("Qwen/Qwen2.5-7B-Instruct", "qwen", 3584,
                      20, "stander_jailbreak_eval_qwen",
                      "hs_qwen_aligned_L20.npz"),
    "llama_aligned": ("NousResearch/Meta-Llama-3.1-8B-Instruct", "llama", 4096,
                      16, "stander_jailbreak_eval",
                      "hs_llama_aligned_L16.npz"),
}
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


def strat_3way(bs_ds, seed=20260520):
    rng = np.random.default_rng(seed); eraser, depcal, test = [], [], []
    for s in sorted(set(map(str, bs_ds))):
        idx = np.where(np.array(list(map(str, bs_ds))) == s)[0]
        rng.shuffle(idx); n = len(idx); a, b = n // 3, 2 * n // 3
        eraser += list(idx[:a]); depcal += list(idx[a:b]); test += list(idx[b:])
    return (np.array(sorted(eraser)), np.array(sorted(depcal)),
            np.array(sorted(test)))


def derive_recipe_head(Hin, y, Hbs, bs_ds):
    """Reconstruct the SAME erased probe as pivot_heldout_remedy.py and collapse
    it to (w_eff, b_eff, tau_dep) acting directly on raw h."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(42)
    ben = np.where(y == 0)[0]; jb = np.where(y == 1)[0]
    rng.shuffle(ben); rng.shuffle(jb)
    cal_b, te_b = ben[:len(ben) // 2], ben[len(ben) // 2:]
    tr = np.array(sorted(list(cal_b) + list(jb[:len(jb) // 2])))
    bs_er, bs_dc, _ = strat_3way(bs_ds)
    src = ["indist_benign"] + sorted(set(map(str, bs_ds[bs_er])))
    # CHECK fix H-1: LEACE fit excludes in-dist test partition te_b.
    Xfit = np.vstack([Hin[cal_b], Hbs[bs_er]])
    zlab = np.array([0] * len(cal_b) +
                    [src.index(str(s)) for s in bs_ds[bs_er]])
    Zoh = np.eye(len(src))[zlab][:, 1:]
    mu, R, rnk = leace(Xfit, Zoh)
    er = lambda X: ((X - mu) @ R.T + mu).astype(np.float32)
    # erased probe
    sc = StandardScaler().fit(er(Hin[tr]))
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(er(Hin[tr])), y[tr])
    # derive (w_eff, b_eff) so sigmoid(w_eff . h + b_eff) == clf.predict_proba(...)[..,1]
    v = (clf.coef_[0] / sc.scale_).astype(np.float32)       # (d,)
    intercept = float(clf.intercept_[0])
    w_eff = (R.T @ v).astype(np.float32)
    b_eff = float((v - w_eff) @ mu) + intercept - float(v @ sc.mean_)
    # deployment tau on the erased calib benign-stress
    score_np = lambda H: 1.0 / (1.0 + np.exp(-(H @ w_eff + b_eff)))
    tau_dep = conformal_tau(score_np(Hbs[bs_dc]))
    return w_eff, b_eff, tau_dep, rnk


def jb_prompts(pred_dir, ntr, nte, seed):
    import pandas as pd
    f = sorted(glob.glob(str(PRED / pred_dir / "*_predictions.csv")))[0]
    df = pd.read_csv(f)
    t = [uc(x) for x in df[df["label"] == 0]["text"]]
    t = [x for x in t if x and len(x) > 8]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(t))
    return [t[i] for i in idx[:ntr]], [t[i] for i in idx[ntr:ntr + nte]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--k", type=int, default=16, help="suffix length")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--n-cand", type=int, default=64)
    ap.add_argument("--train-batch", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-test", type=int, default=512)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    hf, arch, hidden, layer, pdir, hs_fn = MODELS[a.model]

    z = np.load(PIV / hs_fn, allow_pickle=True)
    Hin, y, Hbs, bs_ds = z["Hin"], z["y"], z["Hbs"], z["bs_ds"]
    w_eff_np, b_eff, tau_dep, rnk = derive_recipe_head(Hin, y, Hbs, bs_ds)
    print(f"[recipe-gcg/{a.model}] L{layer} rank={rnk} tau_dep={tau_dep:.4f} "
          f"|w_eff|={np.linalg.norm(w_eff_np):.3f} b_eff={b_eff:.3f}", flush=True)

    tr, te = jb_prompts(pdir, a.n_train, a.n_test, a.seed)
    print(f"  train jb={len(tr)} test jb={len(te)} k={a.k} steps={a.steps}",
          flush=True)

    import torch, torch.nn as nn
    from transformers import AutoTokenizer, Qwen2ForCausalLM, LlamaForCausalLM
    torch.manual_seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}")
    tok = AutoTokenizer.from_pretrained(hf)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    LM = Qwen2ForCausalLM if arch == "qwen" else LlamaForCausalLM
    model = LM.from_pretrained(hf, torch_dtype=torch.float32)
    model.model.layers = nn.ModuleList(model.model.layers[:layer])
    # 1-class linear head = recipe's effective head
    model.lm_head = nn.Linear(hidden, 1, bias=True)
    with torch.no_grad():
        model.lm_head.weight.copy_(torch.from_numpy(w_eff_np[None, :]).float())
        model.lm_head.bias.copy_(torch.tensor([b_eff], dtype=torch.float32))
    model.to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    emb = model.get_input_embeddings()
    W = emb.weight                                   # (V, d)
    V = W.shape[0]

    def tok_prompt(p):
        s = tok.apply_chat_template([{"role": "user", "content": p}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False, return_tensors="pt"
                   )["input_ids"][0]

    # init suffix tokens (random from non-special vocab)
    rng = np.random.default_rng(a.seed)
    forbid = set(tok.all_special_ids)
    pool = [i for i in range(V) if i not in forbid]
    suffix_ids = torch.tensor(rng.choice(pool, a.k, replace=False),
                              dtype=torch.long, device=dev)

    def score_batch(prompts_ids):
        """Forward each prompt+suffix through the truncated backbone, returning
        the recipe SCORE (sigmoid of the 1-class logit at the last token)."""
        scores = []
        for ids in prompts_ids:
            full = torch.cat([ids.to(dev), suffix_ids])
            with torch.no_grad():
                out = model(full[None, :])
                logit = out.logits[0, -1, 0].item()
            scores.append(1.0 / (1.0 + np.exp(-logit)))
        return np.array(scores)

    # Pretokenize prompts
    prompts_tr = [tok_prompt(p)[: a.max_length] for p in tr]
    prompts_te = [tok_prompt(p)[: a.max_length] for p in te]

    # clean recall before attack
    s_te_clean = score_batch(prompts_te)
    recall_clean = float((s_te_clean >= tau_dep).mean())
    print(f"  CLEAN recall @ tau_dep = {recall_clean:.3f}", flush=True)

    # GCG loop — universal, on a random train batch each step
    for step in range(a.steps):
        # pick a random train batch
        batch_idx = rng.choice(len(prompts_tr), a.train_batch, replace=False)
        # compute gradient of mean batch score w.r.t. suffix one-hot embedding
        one_hot = torch.zeros(a.k, V, device=dev, dtype=torch.float32)
        one_hot.scatter_(1, suffix_ids[:, None], 1.0)
        one_hot.requires_grad_(True)
        suffix_emb = one_hot @ W                          # (k, d)
        # build batch
        total_loss = 0.0
        for bi in batch_idx:
            ids = prompts_tr[bi].to(dev)
            x_emb = emb(ids)
            full_emb = torch.cat([x_emb, suffix_emb], 0)[None]
            out = model(inputs_embeds=full_emb)
            score = torch.sigmoid(out.logits[0, -1, 0])
            total_loss = total_loss + score
        total_loss = total_loss / a.train_batch
        grad = torch.autograd.grad(total_loss, one_hot)[0]  # (k, V)
        # pick top-k candidate replacements
        topk_idx = (-grad).topk(a.topk, dim=1).indices    # (k, topk)
        # sample n_cand random (pos, tok) swaps
        cand = []
        for _ in range(a.n_cand):
            p = int(rng.integers(0, a.k))
            t_ = int(topk_idx[p, int(rng.integers(0, a.topk))].item())
            new = suffix_ids.clone(); new[p] = t_
            cand.append(new)
        # evaluate candidates on the SAME batch (greedy)
        best_loss = total_loss.item(); best_suffix = suffix_ids.clone()
        for c in cand:
            with torch.no_grad():
                ls = 0.0
                for bi in batch_idx:
                    full = torch.cat([prompts_tr[bi].to(dev), c])
                    out = model(full[None, :])
                    ls += torch.sigmoid(out.logits[0, -1, 0]).item()
                ls /= a.train_batch
            if ls < best_loss:
                best_loss = ls; best_suffix = c
        suffix_ids = best_suffix
        if step % 10 == 0 or step == a.steps - 1:
            print(f"  step {step+1}/{a.steps} batch loss={best_loss:.4f}",
                  flush=True)

    # final eval
    s_te_adv = score_batch(prompts_te)
    recall_adv = float((s_te_adv >= tau_dep).mean())
    evasion = (recall_clean - recall_adv) / max(recall_clean, 1e-9)
    suffix_text = tok.decode(suffix_ids.cpu().tolist(),
                             clean_up_tokenization_spaces=False)
    res = {a.model: {
        "layer": layer, "leace_rank": int(rnk),
        "k": a.k, "steps": a.steps, "n_train": len(tr), "n_test": len(te),
        "tau_dep": round(tau_dep, 4),
        "recipe_recall_clean@tau_dep": round(recall_clean, 4),
        "recipe_recall_adv@tau_dep":   round(recall_adv, 4),
        "recipe_aware_evasion_rate":   round(evasion, 4),
        "suffix_tokens": [int(t) for t in suffix_ids.cpu().tolist()],
        "suffix_text":   suffix_text[:200]}}
    op = PIV / "adaptive_gcg_recipe.json"
    allr = json.loads(op.read_text()) if op.exists() else {}
    allr.update(res); op.write_text(json.dumps(allr, indent=1))
    print(f"\nRECIPE-AWARE GCG: recall {recall_clean:.3f} -> {recall_adv:.3f} "
          f"(evasion {evasion:+.3f})")
    print(f"written: {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
