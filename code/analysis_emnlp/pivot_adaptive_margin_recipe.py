#!/usr/bin/env python
"""B1' — RECIPE-AWARE ADAPTIVE UPPER BOUND (closed-form L2 margin).

Since the recipe (LEACE-erased probe + deployment conformal) is a SINGLE
LINEAR classifier on the raw hidden state h (see derivation in
pivot_adaptive_gcg_recipe.py: w_eff = R.T @ (coef/scale_), b_eff per the
collapse), the SMALLEST L2 perturbation delta in hidden-state space that flips
a held-out jailbreak below the deployment threshold has the closed form

  delta_min(h) = (w_eff . h + b_eff - logit(tau_dep)) / ||w_eff||_2

(non-negative for points currently above tau_dep). This is the strongest
possible upper bound on a recipe-aware adversarial attack in continuous
hidden-state space; any discrete token-level GCG can at best match it. Same
analysis for the baseline probe gives a directly comparable adaptive
robustness margin. CPU only, cached states.

For each backbone:
  - derive (w_eff_recipe, b_eff_recipe, tau_dep_recipe) per the recipe.
  - derive (w_eff_base,   b_eff_base,   tau_dep_base)   per the baseline probe
    under the SAME deployment-calibration discipline.
  - per held-out jailbreak with current score >= tau_dep, compute delta_min.
  - report median delta_min normalized by typical hidden-state norm (so cross-
    backbone comparable).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
PIV = REPO / "data/emnlp2026/pivot"
CACHES = {"qwen_aligned": "hs_qwen_aligned_L20.npz",
          "llama_aligned": "hs_llama_aligned_L16.npz",
          "mistral_aligned": "hs_mistral_aligned_L16.npz"}
ALPHA = 0.05


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


def derive_linear_head(Htr_in, ytr, scaler_target_Hin):
    """Train LogReg on Htr_in (with optional erasure already applied) and
    collapse to (w_eff, b_eff) acting on the SAME pre-erasure h."""
    sc = StandardScaler().fit(scaler_target_Hin)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Htr_in), ytr)
    v = (clf.coef_[0] / sc.scale_).astype(np.float32)
    intercept = float(clf.intercept_[0])
    return v, intercept, sc.mean_.astype(np.float32)


def logit(p):
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    return float(np.log(p / (1 - p)))


def main():
    out = {}
    for model, fn in CACHES.items():
        z = np.load(PIV / fn, allow_pickle=True)
        Hin, y, Hbs, bs_ds = z["Hin"], z["y"], z["Hbs"], z["bs_ds"]
        rng = np.random.default_rng(42)
        ben = np.where(y == 0)[0]; jb = np.where(y == 1)[0]
        rng.shuffle(ben); rng.shuffle(jb)
        cal_b, te_b = ben[:len(ben) // 2], ben[len(ben) // 2:]
        tr = np.array(sorted(list(cal_b) + list(jb[:len(jb) // 2])))
        te = np.array(sorted(list(te_b) + list(jb[len(jb) // 2:])))
        bs_er, bs_dc, _ = strat_3way(bs_ds)
        src = ["indist_benign"] + sorted(set(map(str, bs_ds[bs_er])))
        # CHECK fix H-1: LEACE fit excludes in-dist test partition te_b.
        Xfit = np.vstack([Hin[cal_b], Hbs[bs_er]])
        zlab = np.array([0] * len(cal_b) +
                        [src.index(str(s)) for s in bs_ds[bs_er]])
        Zoh = np.eye(len(src))[zlab][:, 1:]
        mu, R, rnk = leace(Xfit, Zoh)
        er = lambda H: ((H - mu) @ R.T + mu).astype(np.float32)

        # ---- recipe head (operating on raw h via collapse) ----
        v_r, int_r, scaler_mean_r = derive_linear_head(
            er(Hin[tr]), y[tr], er(Hin[tr]))
        # collapse er o scaler o logreg into one effective head w · h + b
        # logit = v_r . er(h) + int_r - v_r . scaler_mean_r
        #       = v_r . ((h - mu) @ R.T + mu) + ...
        # w_eff = R.T @ v_r ; b_eff = (v_r - w_eff) . mu + int_r - v_r . scaler_mean_r
        w_eff = (R.T @ v_r).astype(np.float32)
        b_eff = float((v_r - w_eff) @ mu) + int_r - float(v_r @ scaler_mean_r)
        score_recipe = lambda H: 1 / (1 + np.exp(-(H @ w_eff + b_eff)))
        # --- CHECK.md C-4: assert closed-form head equals the pipeline scorer.
        # The headline "1.4-2.1x larger delta_min" depends on (w_eff, b_eff)
        # being arithmetically equal to clf.predict_proba(sc.transform(er(.))).
        # Verify on a held-out slice (te) and report max residual.
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        _sc_r = StandardScaler().fit(er(Hin[tr]))
        _clf_r = LogisticRegression(max_iter=2000).fit(_sc_r.transform(er(Hin[tr])), y[tr])
        _s_pipeline = _clf_r.predict_proba(_sc_r.transform(er(Hin[te])))[:, 1]
        _s_collapse = score_recipe(Hin[te])
        _resid_r = float(np.abs(_s_pipeline - _s_collapse).max())
        assert _resid_r < 1e-3, f"recipe head-collapse residual={_resid_r}"
        tau_dep_r = conformal_tau(score_recipe(Hbs[bs_dc]))
        L_tau_r = logit(tau_dep_r)

        # ---- baseline head (no erasure) ----
        v_b, int_b, scaler_mean_b = derive_linear_head(
            Hin[tr], y[tr], Hin[tr])
        # w_base = v_b (no LEACE); b_base = int_b - v_b . scaler_mean_b
        w_base = v_b
        b_base = int_b - float(v_b @ scaler_mean_b)
        score_base = lambda H: 1 / (1 + np.exp(-(H @ w_base + b_base)))
        tau_dep_b = conformal_tau(score_base(Hbs[bs_dc]))
        L_tau_b = logit(tau_dep_b)

        # ---- compute closed-form min-L2 perturbation per held-out jb ----
        jte = te[y[te] == 1]
        H_jb = Hin[jte]
        # for each h with score >= tau_dep, delta_min = (logit_h - L_tau) / ||w||
        norm_w_r = float(np.linalg.norm(w_eff))
        norm_w_b = float(np.linalg.norm(w_base))
        logits_r = H_jb @ w_eff + b_eff
        logits_b = H_jb @ w_base + b_base
        m_r = (logits_r - L_tau_r) / norm_w_r          # signed margin (positive = above tau)
        m_b = (logits_b - L_tau_b) / norm_w_b
        # only consider currently-classified-as-jailbreak (above tau)
        m_r_pos = m_r[m_r > 0]; m_b_pos = m_b[m_b > 0]
        # normalize by typical hidden-state norm for cross-backbone comparison
        h_norm_typ = float(np.median(np.linalg.norm(H_jb, axis=1)))

        out[model] = {
            "leace_rank": rnk,
            "head_collapse_max_residual_recipe": round(_resid_r, 6),
            "tau_dep_recipe":   round(tau_dep_r, 4),
            "tau_dep_baseline": round(tau_dep_b, 4),
            "||w_eff_recipe||_2":   round(norm_w_r, 4),
            "||w_eff_baseline||_2": round(norm_w_b, 4),
            "typical_hidden_norm":  round(h_norm_typ, 4),
            "recipe_jb_above_tau":  int((m_r > 0).sum()),
            "baseline_jb_above_tau": int((m_b > 0).sum()),
            "median_min_delta_L2_recipe":   round(float(np.median(m_r_pos)) if len(m_r_pos) else float("nan"), 4),
            "median_min_delta_L2_baseline": round(float(np.median(m_b_pos)) if len(m_b_pos) else float("nan"), 4),
            "median_min_delta_NORMALIZED_recipe":   round(float(np.median(m_r_pos)) / h_norm_typ if len(m_r_pos) else float("nan"), 4),
            "median_min_delta_NORMALIZED_baseline": round(float(np.median(m_b_pos)) / h_norm_typ if len(m_b_pos) else float("nan"), 4),
        }
        r = out[model]
        print(f"\n[{model}]  rank={rnk}")
        print(f"  ||w_eff|| recipe={norm_w_r:.3f}  baseline={norm_w_b:.3f}  "
              f"typical |h|={h_norm_typ:.1f}")
        print(f"  median min L2 perturbation to evade @ deploy-tau:")
        print(f"    recipe:   {r['median_min_delta_L2_recipe']}  "
              f"(={r['median_min_delta_NORMALIZED_recipe']} of typical |h|)")
        print(f"    baseline: {r['median_min_delta_L2_baseline']}  "
              f"(={r['median_min_delta_NORMALIZED_baseline']} of typical |h|)")
    op = PIV / "adaptive_margin_recipe.json"
    op.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
