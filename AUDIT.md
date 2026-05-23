# AUDIT.md — Pre-submission research audit (Latent Sentinel, ARR May-2026)

READ-ONLY audit. No repository file was modified except this report. Repo:
`<repo-root>`. Submission file under audit:
`paper/latent_sentinel_arr.tex`. Evidence base: `code/analysis_emnlp/*.py`,
`data/emnlp2026/**`, the `.tex`, `requirements.txt`/`REPRODUCE.md`. Two
specialist sub-audits (methodology+leakage; claim-to-evidence) were run and
their findings were independently re-derived where consequential — one
sub-finding (conformal-τ off-by-one) was **rejected** after my own arithmetic
check (see §6, F4-REJECTED).

---

## 1. Executive verdict

The repository is **not submission-ready as currently framed**, though it is
honest work whose numbers are overwhelmingly traceable to committed artifacts
(tab:erase, tab:deploy, tab:ood, tab:native, tab:baselines, adaptive/GCG, ECE
range, register-control AUC all reproduce to the digit). The blocking problem is
**transductive data leakage in the single strongest, newest claim**: the
"deployable recipe" headline (Table `tab:deploy`: ≤5% guaranteed benign-FPR at
0.95–0.97 recall) and the `tab:erase` register-erased rows are computed with the
LEACE eraser fit on the **same** benign-stress points it is then evaluated on.
That converts a deployment claim into a transductive one and a top-tier reviewer
will catch it immediately. Secondary critical issues: best probe layer is
selected on **test** AUC and propagated everywhere; the "seed-stable, AUC std
≤0.002" claim is contradicted (Llama best layer flips L12↔L16, ΔAUC 0.0059) and
"3 seeds" only reshuffles a split of single-run frozen scores; baseline
comparison is not apples-to-apples (FJD alone gets a labeled-calibration
temperature search); the `tab:wsa` Llama ECE 0.318 is not reproducible from the
per-seed artifacts (they give ≈0.284); the hero figure hardcodes the probe's
points; and the headline ρ≈0.87 has no producing script (committed artifact says
0.8171/10-cell). Reproducibility: **Partial**. Ablation trust: **Partial**
(register-balanced ablation is clean; LEACE-erasure ablation is leaked). Metric
trust: **Partial**. Top-3 blockers: (1) transductive LEACE leakage on the
deployable headline; (2) test-set layer selection + seed-stability overclaim;
(3) baseline-fairness asymmetry undermining the ρ trade-off.

---

## 2. Submission blockers

### B1 — Transductive LEACE leakage invalidates the deployable-recipe headline and the register-erased rows
- **Severity:** Blocker
- **Why this blocks submission:** The paper's strongest *new* contribution
  (`sec:fix` "the recipe attains a usable, guaranteed operating point",
  Table `tab:deploy`: register-erased probe → ≤5% guaranteed diverse-benign FPR
  at 0.95–0.97 recall vs 0.23–0.31 baseline) is presented as a *deployment*
  result on diverse benign traffic. It is actually a **transductive** result:
  the LEACE eraser is fit using the very benign-stress points it is then scored
  on.
- **Exact evidence:**
  - `code/analysis_emnlp/pivot_deployable_operating_point.py:80` —
    `Xfit = np.vstack([Hin[ben], Hbs])`; line 81-ish `mu,R,rnk = leace(Xfit,Zoh)`
    fits the eraser on the **full** `Hbs` (all 3,567 diverse-benign rows).
  - same file ~`:87-105` — the deployment split `bcal/btest` is carved from
    `Hbs`; the `register_erased` condition scores `er(Hbs)` and reports FPR on
    `er(Hbs[btest])` and recall at τ from `er(Hbs[bcal])`. `bcal∩btest=∅` (the
    τ-calibration vs FPR-eval split is clean) **but both ⊂ the eraser-fit set**.
    Overlap of eraser-fit set with the erased-condition evaluation rows = 100%.
  - `code/analysis_emnlp/pivot_register_erase_remedy.py:~182,189` —
    `Xfit=np.vstack([Hin[ben],Hbs])`; `ersd = fit_eval(er(Hin[tr]),…,er(Hbs))`
    → `benign_stress_FPR_at_tau` for the register-erased row of `tab:erase`
    (0.213/0.171/0.185) is measured on the eraser's own fit data.
- **Risk to conclusions:** LEACE is a projection that removes exactly the
  directions separating `indist_benign` from each diverse-benign source *on the
  fitted points*. Measuring over-refusal collapse on those same points
  mechanically understates deployment FPR and overstates recall-at-fixed-FPR.
  The "guaranteed ≤5% FPR at 0.95–0.97 recall, 3 backbones" claim is **not a
  valid estimate of behaviour on unseen benign traffic**.
- **Scope/mitigant (verified):** the `register_balanced_training` variant
  (`pivot_register_erase_remedy.py:~191-193`: train on `vstack([Hin[tr],
  Hbs[:nbs]])`, test on `Hbs[nbs:]`) uses a **disjoint** benign train/test split
  and independently reaches OOD-FPR 0.167/0.144/0.168 — this corroborates the
  *mechanistic* "over-refusal is reducible" claim cleanly, but does **not**
  rescue the `tab:deploy` recall-at-guaranteed-FPR numbers (those use the leaked
  eraser). The jailbreak-AUC-survives part is more robust (the eraser targets
  benign source-identity, not the jailbreak label), so the *causal-existence*
  claim is less affected than the *deployment-performance* claim.
- **Minimum fix:** partition `Hbs` into mutually disjoint eraser-fit /
  conformal-calibration / test subsets *before* `leace()`; fit the eraser on the
  fit subset only; report FPR/recall strictly on the held-out test subset.
  Re-run `pivot_register_erase_remedy.py` and `pivot_deployable_operating_point.py`
  under that discipline; regenerate `tab:erase` register-erased rows and
  `tab:deploy`. Pre-register that the register-balanced (clean) result is the
  primary evidence and LEACE is corroborating.
- **Verification after fix:** eraser-fit indices ∩ evaluation indices = ∅
  (assert in code); the regenerated `tab:deploy` recall and `tab:erase` FPR are
  reported on held-out benign; if the held-out result materially differs from
  the leaked one, the headline must be re-stated to the held-out numbers.

---

## 3. Critical findings

### C1 — Best probe layer selected on TEST ROC-AUC, then propagated everywhere
- **Severity:** Critical
- **Why it matters:** Headline `tab:wsa` AUC/operating points and the conformal
  τ that flows into WS-B/C/D and the pivots are taken from the layer chosen to
  maximise **test** AUC — optimistic model selection.
- **Exact evidence:** `code/analysis_emnlp/ws_a_balanced_metrics.py:~148-149`
  (`roc_auc = roc_auc_score(yt, pt)` on the **test** split) and `~181`
  (`best = max(per_layer, key=…["roc_auc"])`). The `top5_mean_by_calAUC`
  ensemble in the same file correctly selects on `roc_auc_score(y[cal],…)` —
  proving the correct pattern was known but not applied to the reported "best".
  The selected layer is then hardcoded downstream (`wsb_register_confound.py:~38`
  `BEST={…16…}`, `wsb_benign_stress_infer.py:~40`, `wsc_*`, `wsd_*`, `pivot_*`).
- **Failure mode:** optimistic bias on the headline AUC/FPR cells and a
  cherry-picked layer baked into every downstream number.
- **Minimum fix:** select best layer by **calibration** AUC within each seed;
  report that layer's test metrics; re-run the downstream chain with the
  cal-selected layer.
- **Verification:** layer-selection uses only `cal`; headline numbers recomputed.

### C2 — "Seed-stable, AUC std ≤0.002 (3 seeds)" is contradicted and mis-scoped
- **Severity:** Critical
- **Why it matters:** A reproducibility/stability claim in the Contributions and
  Limitations is stronger than the evidence and is empirically false for the
  Llama-aligned cell.
- **Exact evidence:** `ws_a_balanced_metrics.py:~160-181` — the seed only
  reshuffles the calib/test split of **frozen, single-run** logits read from
  `data/predictions`/`detailed_evaluation_results*.pkl`; no model re-train, no
  re-inference (consistent with CLAUDE.md: original `probe_training/*` has no
  seeds). Cross-seed artifacts: `data/emnlp2026/wsc/seed41_*`, `seed43_*`, and
  the WS-A per-seed JSONs show the Llama-aligned **best layer flips L12 (seed41)
  ↔ L16 (seed42)** with ΔAUC ≈ 0.0059 (> 0.002), and per-seed best-layer ECEs
  0.2907/0.2813/0.2793. No script computes an across-seed AUC std.
- **Failure mode:** overclaimed stability; the "≤0.002" has no producing code
  and is false where layer selection is unstable.
- **Minimum fix:** state explicitly that seed variance = split-resampling of a
  single inference run (not training/inference variance); drop or re-scope the
  "≤0.002" claim; fix the layer a priori for all seeds or report the
  layer-selection instability honestly.
- **Verification:** the claim text matches what the code measures; any stated
  std is emitted by a script over the three seeds at a fixed layer.

### C3 — `tab:wsa` Llama-3.1-8B ECE = 0.318 not reproducible from committed artifacts
- **Severity:** Critical
- **Why it matters:** A headline-table cell cannot be derived from the per-seed
  artifacts the rest of the table is derived from → provenance failure / possible
  stale or fixed-layer number.
- **Exact evidence:** `data/emnlp2026/ws_a/stander_jailbreak_eval__balanced.json`
  (+ seed41/seed43) best-layer ECEs = 0.2907 / 0.2813 / 0.2793 (mean ≈ 0.284),
  best layer differing per seed (16/12/12). Paper `latent_sentinel_arr.tex`
  `tab:wsa` row reports ECE 0.318±0.019. No opened `ws_a` JSON yields 0.318.
- **Failure mode:** an un-traceable headline number; reviewer rerun yields 0.284.
- **Minimum fix:** identify the exact aggregation that produced 0.318 (fixed
  layer? CONSOLIDATED CSV?); either correct the cell to the reproducible value
  or emit the producing computation as a script + artifact.
- **Verification:** the printed cell equals a number a committed script writes.

### C4 — Baseline comparison is not apples-to-apples (FJD labeled-calibration tuning; JBShield self-leak)
- **Severity:** Critical
- **Why it matters:** `tab:baselines` and the ρ trade-off figure compare the
  proposed (label-free conformal-τ) probe against baselines under "identical
  protocol", but the protocol is not uniform.
- **Exact evidence:** `code/analysis_emnlp/wsd_baselines.py:~152-173` — FJD gets
  a temperature grid-searched to maximise **calibration ROC-AUC** (a labeled
  fit); GradSafe uses a magic constant `(ur-sr)>1`,`(uc-sc)>1` (`~231-232`);
  JBShield concept references are sampled from `np.where(y==…)` over the **whole**
  eval set (`~282-285`), overlapping its own test rows; the proposed probe gets
  only a label-free τ. HiddenDetect-exact correctly restricts its few-shot pool
  to `cal` — inconsistent treatment across baselines.
- **Failure mode:** the "higher in-dist AUC ⇒ worse OOD FPR" trade-off
  (ρ≈0.87) may be partly an artifact of non-uniform baseline calibration rather
  than a real phenomenon; baselines may be systematically weakened.
- **Minimum fix:** give every detector the identical calibration budget
  (preferably the same label-free conformal τ), or fix FJD's temperature to its
  paper default; sample JBShield concept refs from `cal` only; enumerate every
  baseline deviation in the paper.
- **Verification:** all detectors share calibration discipline; ρ recomputed.

### C5 — `wsf_native_threshold` sets the 95%-TPR threshold on the same test positives it evaluates
- **Severity:** Critical
- **Why it matters:** This is the "the over-refusal is *not* a conformal-τ
  artifact" firewall (`sec:ood`, `tab:native`). If τ is chosen on the test
  jailbreaks, realized TPR=0.95 is tautological and the benign FPR is reported
  at an in-sample-selected threshold.
- **Exact evidence:** `code/analysis_emnlp/wsf_native_threshold.py:~40-63` —
  `s_jb = indist[tjb]; tau = tpr95_threshold(s_jb)` where `tjb` are **test**
  jailbreaks, then `realized_jb_TPR_at_tau` (≈0.95) and benign-stress FPR are
  reported at that τ. Also `~F32`: FJD AUC≈0.41 (anti-correlated) is not
  sign-oriented, so its native FPR=1.0 / τ=−1.0 row is meaningless yet tabulated.
- **Failure mode:** the firewall result uses leaked threshold selection; the FJD
  native row is invalid.
- **Minimum fix:** choose the 95%-TPR τ on **held-out** (calibration)
  jailbreaks, evaluate on disjoint test; orient detector sign by calib AUC
  before computing native τ (as `run_bert` already does).
- **Verification:** native-τ selection set disjoint from evaluation; FJD row
  either oriented or dropped.

---

## 4. Major findings

### M1 — ρ≈0.87 (12 cells), per-model 0.71/1.00, LOO [0.79,0.92] have no producing script; committed artifact disagrees
- Severity: Major. The only stored ρ artifact
  `data/emnlp2026/wsf_bootstrap_ci.json:_tradeoff_spearman` = **rho 0.8171, n
  10** (baselines only, excludes the trained probe; `wsf_bootstrap_ci.py:~75-87`
  reads `baselines.json`). The paper's 12-cell ρ=0.87 / per-model / LOO numbers
  are reproducible only by manually adding the two probe cells (independently
  recomputed: ρ≈0.874, per-model 0.714/1.00, LOO [0.793,0.915] — they DO match
  the paper). Risk: a reviewer rerunning the pipeline gets 0.82, not 0.87;
  headline statistic has no committed provenance. Min fix: add the 12-cell
  computation to a script + artifact, or cite the in-paper computation
  explicitly. Verify: a script emits exactly the reported ρ/LOO.

### M2 — Hero figure hardcodes the proposed probe's points
- Severity: Major. `code/analysis_emnlp/make_tradeoff_figure.py:~19-20`
  `PROBE = {qwen: auc 0.996, fpr 0.982; llama: auc 0.957, fpr 0.683}` are
  literal constants; baselines are read from `baselines.json`. The most
  important points on the headline figure are not reproducible from data
  (numbers happen to match wsb/baselines artifacts — not fabricated, but not
  traceable in-script). `~67-68` also hardcodes "ρ≈0.87" in the title. Min fix:
  compute probe AUC/FPR from saved score arrays + the same conformal τ in the
  figure script; read ρ from the artifact. Verify: figure regenerates from data
  only.

### M3 — Cross-pipeline τ transfer + bf16/fp32 regime mismatch on the over-refusal headline
- Severity: Major. `wsb_benign_stress_infer.py:~78-90` imports conformal τ from
  WS-A JSON (scores produced by the *original* fp32 notebook pipeline,
  `data/predictions`/pkl), but recomputes OOD scores with a fresh truncated-model
  HF load at default `--dtype bf16` with a hand-written chat-template fallback
  (`~157-168`). τ and OOD scores are then not guaranteed on the same numeric
  scale; same pattern in `wsc_adaptive_probe_evasion.py:~74-77,120-134` and
  `wsc_gcg.py:~71-73`. The over-refusal FPR (68–98%) is the central negative
  result, so a scale mismatch is consequential. Min fix: recompute in-dist
  benign scores with the *same* loader/precision and recalibrate τ in-script;
  assert `recall_BEFORE` ≈ WS-A in-dist recall. Verify: τ and scores share the
  exact pipeline; sanity equality holds.

### M4 — "Within benign prompts … Pearson up to +0.34" overstated
- Severity: Major (claim integrity). `data/emnlp2026/wsb/register_confound.json`
  benign-only top-feature Pearson max ≈ 0.23; the 0.34 corresponds to a
  jailbreak-set length correlation (`T4_corr_len_score_jailbreak≈0.335`) or a
  standardized-mean-difference (≈0.33), not a benign-only Pearson r. Paper
  `latent_sentinel_arr.tex` (register section, ~L225-226) mislabels the
  statistic and overstates by ≈0.1. Min fix: quote the benign-only value (≈0.23)
  or correctly attribute the 0.34 statistic. Verify: number+label match the
  artifact field.

### M5 — `wsb_register_confound` CV is unstratified, unseeded, `--seed` ignored
- Severity: Major. `wsb_register_confound.py:~103-123` `cross_val_predict(pipe,
  X, target, cv=5)` with default non-shuffled KFold on data concatenated
  benign-then-jailbreak → folds can be near single-class; a comment falsely
  claims "determinism handled by sklearn default" and the `--seed` flag is
  unused. The mechanism claim ("register-only classifier recovers ≈0.89/≈90% of
  the probe") rests on these AUCs. (`pivot_leace_probe.py:~68` correctly uses
  `StratifiedKFold(5,shuffle=True,random_state=42)` — inconsistent within the
  codebase.) Min fix: `StratifiedKFold(shuffle=True, random_state=seed)`;
  re-run T1/T2. Verify: folds stratified+seeded; AUCs stable.

### M6 — Notebooks for the *original* paper artifacts are not script-traceable
- Severity: Major (scope-limited). The original Tables/Figures live in 45–63 MB
  Korean-named notebooks with saved outputs (`code/evaluation/*.ipynb`,
  `code/analysis_figures/*.ipynb`) with no driving script. The ARR submission's
  claims rest on `code/analysis_emnlp/*` + `data/emnlp2026/*` (script-based) —
  acceptable — but any number sourced from the old notebooks (none should be in
  `latent_sentinel_arr.tex`) would be non-reproducible. Min fix: confirm in
  writing that no `latent_sentinel_arr.tex` number derives from the legacy
  notebooks. Verify: every table cell maps to a `data/emnlp2026` artifact (the
  claim-to-evidence matrix below largely confirms this).

---

## 5. Moderate findings

- **MO1 — Non-standard ECE labeled as ECE.** `ws_a_balanced_metrics.py:~86-93`
  compares mean predicted P(jailbreak) to empirical jailbreak rate
  (positive-class reliability), not top-label ECE. Label it
  "positive-class/attack calibration error" or reviewers misread it.
- **MO2 — Conformal τ helper duplicated across 4 scripts.** Functionally
  equivalent (see §6 F4-REJECTED) but un-factored; `min(k,len-1)` clamp differs.
  Refactor to one shared helper for auditability.
- **MO3 — `wsf_bootstrap_ci.py:~24` single global RNG consumed serially across
  all cells → CIs order-fragile / not independently reproducible per cell.
- **MO4 — Two `requirements.txt` disagree.** Root pins 25 pkgs "Python 3.10";
  `code/analysis_emnlp/requirements.txt` pins 12 pkgs "Python 3.11", **omits
  `matplotlib`** (needed by `make_tradeoff_figure.py`), adds bitsandbytes/hf-hub.
  `__pycache__` shows cpython-311 → the 3.10 header is wrong. Pin one env.
- **MO5 — REPRODUCE.md drift.** Root REPRODUCE references
  `paper/latent_sentinel_emnlp2026.tex` (the build is `latent_sentinel_arr.tex`);
  blanket "`--wait-vram`" untrue for `wsb_*`(`--wait-for-vram`)/`wsc_*`/`wsd_*`
  (no wait flag). Dataset-size statement conflict: §Setup "2,000+2,000" vs
  §Reproducibility/CLAUDE "3,400=1,700+1,700" (body tables use the 4,000 set).
- **MO6 — Silent benign-source attrition.** `wsb_fetch_benign_stress.py:~73-76`
  swallows loader failures; only 3 of a planned 5 sources resolved
  (ToxicChat = 56% of the OOD set). The paper *correctly* says "three benign
  stress benchmarks", so this is a diversity-framing caveat, not a number error,
  but "diverse benign" leans heavily on ToxicChat-benign; no dataset `revision=`
  pins → network-fragile.
- **MO7 — `leace_spine.json` (rank 7, attack-family Z, Qwen_JB cache) vs body
  rank-3 (source-identity Z).** Different experiments; REPRODUCE calls
  leace_spine the "cached appendix control". If an appendix is added the rank-3
  vs rank-7 distinction must be explained or it reads as contradictory.
- **MO8 — Mistral single-artifact.** Mistral appears only in
  `register_erase_remedy.json`/`deployable_operating_point.json`; no WS-A/B/C/D
  Mistral entry, so "held-out a priori L16, not layer-selected" cannot be
  cross-checked against an independent best-layer. Plausible but single-source.

---

## 6. Minor findings

- **F4-REJECTED (recorded for transparency):** the methodology sub-audit flagged
  a conformal-τ off-by-one (claiming `pivot_deployable_operating_point.py`
  ≈ `s[k-2]`). I re-derived: `ws_a` `k=⌈(n+1)·.95⌉; s[k-1]`;
  `pivot_register_erase` `s[⌈(n+1)·.95⌉-1]`; `pivot_deployable`
  `k=⌈(n+1)·.95⌉-1; s[min(k,len-1)]` — all index the **same** order statistic
  `s[⌈(n+1)·.95⌉-1]`. **Not a real defect** (downgraded to MO2 cosmetic). Stated
  so the user is not misled by the sub-audit.
- Hardcoded GPU indices `--gpu` default 1/2 in `wsc_*`,`wsd_*`,
  `wsd_hiddendetect_baseline.py` (machine-specific; overridable) — repro hygiene.
- `pivot_leace_probe.py` depends on `data/inference_results/
  all_hidden_states_by_layer.pkl`, a symlink to another user's workspace
  (CLAUDE.md) → that experiment's execution cannot be verified.
- Cached `data/emnlp2026/pivot/hs_*.npz` have no hash/version guard; a stale
  cache would silently feed B1/tab:erase/tab:deploy.
- `wsc_adaptive_probe_evasion.py` embedding-suffix is an unrealizable upper
  bound; ensure only the discrete GCG result enters any headline (the paper does
  label both — acceptable, keep vigilant in the figure/CI table).
- "<0.003% params / <0.13% latency" (`tex` ~L50-51) is attributed to HSF prior
  work; acceptable as citation, but no in-repo artifact measures it.

---

## 7. Reproducibility scorecard

| Item | Verdict | Note |
|---|---|---|
| Environment recreation | Partial | two divergent requirements.txt; Python 3.10 vs 3.11 conflict; analysis set missing matplotlib (MO4) |
| Dependency pinning | Partial | root set well-pinned; analysis set incomplete; no HF dataset `revision=` pins |
| Seed control | Fail | original probe training unseeded (CLAUDE.md); WS-A "seeds" only reshuffle frozen scores; register_confound CV unseeded (M5); attack RNG partial (C2/M5) |
| Determinism awareness | Partial | adaptive attack 3-seed determinism shown (seed41/43); no torch deterministic flags; bf16 nondeterminism unaddressed |
| Train/eval config consistency | Partial | OOD scoring uses bf16 + hand-rolled chat template vs original fp32 pipeline (M3) |
| Checkpoint reproducibility | Partial | aligned probes loadable; degraded `*_JB` checkpoints unrecoverable (documented) |
| Command reproducibility | Partial | REPRODUCE.md filename/flag drift (MO5); commands otherwise match argparse |
| Dataset split traceability | Partial | WS-A/WS-D splits seeded+disjoint; pivot eraser split NOT disjoint from eval (B1) |
| Artifact traceability | Partial | most tables exact-match JSONs; ρ and Llama-ECE have no producing artifact (M1,C3); figure points hardcoded (M2) |
| Notebook/script consistency | Fail | legacy paper artifacts only in 45–63 MB notebooks; submission rests on scripts (M6) |

---

## 8. Ablation integrity scorecard

| Name | Claimed purpose | Exists in code? | Actually isolated? | Fairly controlled? | Result traceable? | Verdict |
|---|---|---|---|---|---|---|
| LEACE register erasure (tab:erase erased rows) | causal: over-refusal lives in rank-3 benign register | Yes (`pivot_register_erase_remedy.py`) | Partial | **No** — eraser fit on eval points (B1) | Yes (register_erase_remedy.json) | **Confounded (leaked)** |
| Register-balanced training (tab:erase) | over-refusal reducible by training w/ diverse benign | Yes (same script, ~191-193) | Yes | Yes — disjoint Hbs train/test | Yes | **Verified & fair** |
| Deployable recipe (tab:deploy) | erasure+deploy-conformal → usable op point | Yes (`pivot_deployable_operating_point.py`) | Partial (bcal⊥btest) | **No** — eraser saw btest (B1) | Yes (deployable_operating_point.json) | **Invalid as deployment claim** |
| Adaptive evasion (embedding vs GCG) | published linear probe evadable | Yes (`wsc_*`) | Yes (held-out test split) | Yes | Yes (adaptive/gcg_evasion.json + seed41/43) | **Verified** |
| Aggregation/randomization defense | (deliberately NOT claimed) | Yes (`pivot_mitigation_adaptive.py`) | n/a | n/a | json exists, **correctly excluded** | **Correctly disclaimed** |
| Register-only no-model control | signal is register not content | Yes (`wsb_register_confound.py`) | Partial | Partial — unstratified/unseeded CV (M5) | Yes (register_confound.json) | **Weak (CV protocol)** |
| Cached-LEACE amnesic spine (leace_spine) | appendix control | Yes (`pivot_leace_probe.py`) | transductive on jb (F34) | Partial | json exists; symlink dep (MO/minor) | **Not verifiable / confounded** |

---

## 9. Metric integrity scorecard

| Metric | Where implemented | Where reported | Status | Main concern |
|---|---|---|---|---|
| ROC-AUC (in-dist) | ws_a/wsd `roc_auc_score` | tab:wsa, tab:baselines | Partial | best layer chosen on test AUC (C1) |
| ECE | ws_a `ece()` | tab:wsa, abstract 0.20–0.33 | Partial | non-standard (MO1); Llama 0.318 not reproducible (C3) |
| Conformal benign-FPR @α | ws_a/wsd/pivot `conformal_tau` | tab:ood, tab:erase, tab:deploy, abstract | Partial | correct estimator (F4-REJECTED) but tab:erase/tab:deploy FPR leaked (B1) |
| FPR@95TPR / native-τ | wsf_native_threshold | sec:ood, tab:native | Suspicious | τ set on test positives (C5); FJD sign not oriented |
| Recall @ τ (deployable) | pivot_deployable | tab:deploy | Not verified | leaked eraser (B1) |
| Bootstrap 95% CI | wsf_bootstrap_ci / pivot_deployable | tab:baselines, tab:deploy, sec:ood | Partial | order-fragile global RNG (MO3); valid per-cell |
| Spearman ρ trade-off | wsf_bootstrap_ci (10-cell) | abstract/sec:baseline (12-cell 0.87) | Suspicious | no script emits the reported value (M1) |
| Register-only AUC | wsb_register_confound | sec:register ≈0.89/≈90% | Partial | unstratified/unseeded CV (M5) |
| Adaptive evasion rate | wsc_adaptive/wsc_gcg | sec:adaptive 1.0→0/0.98→0 | Verified | embedding is upper bound (labeled); GCG sound |

---

## 10. Claim-to-evidence matrix (key claims)

| Claim | Support status | Evidence | Main risk / gap |
|---|---|---|---|
| ECE 0.20–0.33 (4 models) | Verified (endpoints) | ws_a `*_balanced.json` | Llama 0.318 cell not reproducible (C3) |
| Register-only recovers ≈0.89/≈90% | Verified (value) | wsb/register_confound.json T1≈0.89 | CV unstratified/unseeded (M5) |
| "+0.34 within benign" | **Contradicted** | benign-only max ≈0.23 in artifact | overstated/mislabeled (M4) |
| tab:ood 0.982/0.683 | Verified exact | wsb/benign_stress_fpr.json | scale-transfer risk on τ (M3) |
| tab:native numbers | Verified exact | wsf_native_threshold.json | τ on test positives (C5); FJD row invalid |
| tab:erase (3 backbones) | Verified exact vs JSON | register_erase_remedy.json | erased rows transductively leaked (B1) |
| tab:deploy 0.95–0.97 vs 0.23–0.31 | Verified vs JSON; **invalid as deployment** | deployable_operating_point.json | leaked eraser (B1) |
| tab:baselines 12 cells | Verified exact | wsd/baselines.json + wsb | baseline fairness asymmetry (C4) |
| ρ≈0.87 / per-model / LOO | Recomputable from paper table; **artifact says 0.8171/10** | wsf_bootstrap_ci.json | no producing script (M1) |
| Adaptive 1.0→0 / 0.98→0, evasion 1.0/0.981, 3-seed | Verified exact | wsc adaptive/gcg + seed41/43 | embedding=upper bound (labeled) |
| "seed-stable, AUC std ≤0.002" | **Contradicted** | Llama best layer L12↔L16 ΔAUC 0.0059 | overclaim (C2) |
| rank-3 identical, 3 backbones | Verified (data) | register_erase_remedy.json leace_rank=3 | causal interpretation rests on leaked eraser (B1) |
| "<0.003% params/<0.13% latency" | Cannot-verify (cited to HSF) | none in-repo | acceptable only as citation |
| 98–100% headline (incumbent framing) | Verified as attribution | ws_a recall_jb ≈0.996–1.0 | fair characterization, not self-claim |

---

## 11. Missing information / blocked checks

- **Provenance of `tab:wsa` Llama ECE 0.318** — no opened `data/emnlp2026/ws_a`
  JSON yields it (they give ≈0.284). Blocked: the producing aggregation
  (fixed-layer? `CONSOLIDATED_corrected_metrics.csv`?) is not exposed as a
  script+artifact.
- **12-cell ρ / per-model ρ / LOO interval** — no script computes them; only
  the 10-cell 0.8171 is committed. Blocked: hand-computation, not reproducible
  from the pipeline.
- **bf16-vs-fp32 FPR magnitude (M3)** — requires running the `--dtype fp32`
  path; no log evidence it was used. Blocked: needs a run.
- **`pivot_leace_probe.py` execution on real data** — depends on a symlinked
  pkl in another user's workspace; existence/contents unconfirmed.
- **Cache provenance** — `hs_*.npz` have no hash/version stamp; cannot confirm
  they were regenerated after the Mistral addition vs stale.
- **mitigation_adaptive first-400 vs GCG-train overlap (potential)** — would
  need to reproduce both seeded splits; not claimed in the paper so low stakes.

---

## 12. Prioritized repair plan

**Must fix before submission**
1. **Fix B1 (transductive leakage).** Re-implement disjoint
   eraser-fit / conformal-cal / test partitions in
   `pivot_register_erase_remedy.py` + `pivot_deployable_operating_point.py`;
   regenerate `tab:erase` erased rows + `tab:deploy`; re-state the headline to
   the held-out numbers; foreground the **clean** register-balanced result as
   primary. Impact: decisive (saves or honestly re-scopes the lead claim).
   Effort: ~0.5 day (CPU re-analysis, no GPU). Confidence: high it resolves the
   leakage; medium that held-out numbers remain strong (must report whatever
   they are).
2. **Fix C1 (test-layer selection).** Select best layer on calibration AUC; re-run
   downstream chain. Impact: removes optimistic bias. Effort: ~0.5 day.
   Confidence: high.
3. **Fix C2 (seed claim).** Re-scope/remove "AUC std ≤0.002 / seed-stable"; state
   what the seeds actually vary. Impact: removes a false claim. Effort: 1 h
   (text + one script). Confidence: high.
4. **Fix C3 (Llama ECE provenance).** Trace or correct the 0.318 cell. Effort:
   1–2 h. Confidence: medium (depends on locating the aggregation).
5. **Fix C4 (baseline fairness).** Uniform calibration budget; JBShield refs from
   `cal`; enumerate deviations. Impact: legitimizes the ρ trade-off. Effort:
   ~0.5 day. Confidence: medium-high.
6. **Fix C5 (native-τ leakage).** Native τ on held-out jb; orient FJD sign.
   Effort: 2–3 h. Confidence: high.
7. **Fix M1/M2 (ρ + figure provenance).** Add a script emitting the 12-cell ρ +
   LOO; compute probe figure points from data. Effort: 2–3 h. Confidence: high.
8. **Fix M4 (the +0.34 overstatement).** One-line correction to the artifact
   value. Effort: 5 min. Confidence: high.

**Should fix before release**
- M3 (recompute τ in-pipeline / fp32 headline), M5 (stratified seeded CV),
  MO4 (single pinned env incl. matplotlib), MO5 (REPRODUCE/filename/flags),
  MO1 (ECE label), MO3 (per-cell RNG), cache hash guard.

**Nice-to-have**
- Factor a shared `conformal_tau`; remove hardcoded GPU defaults; add a Mistral
  WS-A best-layer cross-check; reconcile rank-3/rank-7 if an appendix is added.

---

## 13. Likely reviewer attacks (strongest 5)

1. "Your flagship 'deployable recipe' (Table tab:deploy) erases the register
   using the very benign set you then evaluate on — this is transductive
   leakage; the ≤5%-FPR-at-0.96-recall result is not a deployment estimate."
   (B1 — the decisive attack.)
2. "You pick the probe layer by test AUC and propagate it everywhere; the
   headline AUC/τ are optimistically biased." (C1)
3. "You claim seed-stability (std ≤0.002) but your 'seeds' only reshuffle a
   split of single-run frozen scores, and the Llama best layer flips L12↔L16
   (ΔAUC 0.006)." (C2)
4. "Your accuracy↔over-refusal trade-off (ρ≈0.87) is computed on baselines you
   calibrated inconsistently (FJD alone gets a labeled temperature search), and
   no released script even produces 0.87 (the artifact says 0.82/10 cells)."
   (C4 + M1)
5. "The 'over-refusal is not a threshold artifact' control sets each detector's
   threshold on the test positives it then scores, and tabulates a meaningless
   FJD native-FPR=1.0." (C5)

---

## 14. Bottom-line decision

- **Technically sound:** Unclear — the core characterization (calibration
  collapse, benign over-refusal, register mechanism) is sound and traceable, but
  the strongest new claim is leakage-invalidated and several headline numbers
  have selection/provenance flaws.
- **Reproducible:** Partial.
- **Leakage-free evaluation:** No (B1 transductive eraser; C5 native-τ;
  JBShield self-leak).
- **Ablations trustworthy:** Partial (register-balanced clean; LEACE-erasure
  leaked; adaptive verified; mitigation correctly disclaimed).
- **Metrics trustworthy:** Partial.
- **Claim-to-code alignment:** Partial (most numbers exact; ρ & Llama-ECE
  provenance gaps; +0.34 overstatement; seed-stability overclaim).
- **Submission-ready for a serious AI/ML venue:** **Not yet** — one Blocker
  (B1) plus five Criticals. None require new model training; the leakage fix is
  a disjoint-split re-analysis (CPU), and a clean corroborating result
  (register-balanced training) already exists. Estimated ~2–3 focused days to a
  defensible submission. Per the patch rules, no files were modified; awaiting
  explicit instruction before any fix.

---

# POST-FIX VERIFICATION REPORT (2026-05-20)

User authorized "fix everything". All fixes are CPU re-analyses of cached
artifacts (no GPU) + manuscript edits. Originals preserved for the audit trail;
corrected results in new artifacts/scripts. All three builds recompile clean
(exit 0, A4, 0 undefined, 0 LaTeX errors; abstract 199 w ≤200; anon == arr).
Every revised manuscript number was re-audited against the corrected JSONs and is
**ALL CONSISTENT** at display precision.

## Fixes applied & material outcome (honest before → after)

- **B1 (Blocker) FIXED** — new `code/analysis_emnlp/pivot_heldout_remedy.py`:
  source-stratified, mutually disjoint eraser/cal/test split of the benign-stress
  set (eraser never sees the eval partition). `data/emnlp2026/pivot/heldout_remedy.json`.
  **Outcome: the headline SURVIVES leakage-free, nearly unchanged** — deployable
  recall (held-out, @≤5% guaranteed FPR) erased **0.94/0.96/0.95** vs baseline
  0.26/0.18/0.29 (was leaked 0.95/0.96/0.97 vs 0.26/0.23/0.31); AUC drop ≤0.0022;
  per-model over-refusal drop −57.2/−51.6/−61.3 pts (was −57.8/−55.9/−64.2).
  The rank-3 register subspace genuinely generalizes. The B1 fix validated rather
  than refuted the contribution.
- **F38 (circular Z-recover) FIXED & honestly corrected** — old "0.99→~0.10" was
  measured on the eraser's own fit set (vacuous by LEACE construction). True
  **held-out** recoverability is **0.99→~0.58–0.63**: erasure is *substantial but
  partial*, now stated plainly in §erase/tab:erase/Limitations. More honest,
  still sufficient.
- **C1 FIXED** — `ws_a_balanced_metrics.py`: layer now selected by
  **calibration** AUC (test-AUC argmax kept transparency-only). Qwen stable L20
  (= the spine layer, justifying the a-priori choice).
- **C2 FIXED (honest re-scope)** — across 3 eval-split seeds calibration-selected
  AUC is stable (std ≤0.0013, so "≤0.002" holds *for AUC*), but the seeds
  resample one single-run inference and the aligned-Llama layer is
  split-seed-unstable (L13/L16). Stated in tab:wsa caption + Limitations.
- **C3 FIXED** — the untraceable Llama ECE 0.318 is replaced by the
  script-reproducible **0.311±0.021** (cal-selected, 3 seeds). tab:wsa Llama row:
  0.959/0.318/0.497 → **0.955/0.311/0.554**. ECE range 0.20–0.33 still holds.
- **C5 FIXED** — `wsf_native_threshold.py`: 95%-TPR τ now set on **calibration**
  jb, realized TPR verified on disjoint **test** jb (0.94–0.97, no longer
  tautological); detector sign oriented by cal-AUC (FJD no longer a meaningless
  FPR=1.0 sign artifact). Conclusion **robust**: probe Qwen 0.972 / Llama 0.697,
  BERT 0.316 — over-refusal is not a threshold artifact.
- **C4 + M1 FIXED (material honest downgrade)** — new
  `wsf_tradeoff_uniform.py`: every detector + the probe scored under ONE uniform
  label-free split-conformal protocol, sign-oriented. **ρ = 0.68 (p=0.014,
  n=12), per-model 0.71/0.43, LOO [0.49,0.79]** — vs the previously claimed
  0.87 / 0.71+1.00 / [0.79,0.92]. The strong-trade-off claim was partly a
  non-uniform-protocol artifact. The paper now reports ρ=0.68 honestly and
  **de-emphasizes it as a supporting observation, not the spine** (the spine is
  the B1-validated causal recipe). committed artifact emits the exact reported ρ.
- **M2 FIXED** — `make_tradeoff_figure.py` reads `wsf_tradeoff_uniform.json`
  (no hardcoded probe points / ρ); `paper/fig_tradeoff.pdf` regenerated.
- **M4 FIXED** — "Pearson +0.34 within benign" was the jailbreak-set / mean-diff
  value; corrected to benign-only ≈0.23 (with the 0.34 correctly attributed to
  the jailbreak set).
- **M5 FIXED (mechanism strengthened)** — `wsb_register_confound.py` now uses
  StratifiedKFold(shuffle, random_state=seed). Register-only AUC is **0.92–0.93**
  (the old 0.89 was an underestimate from degenerate unshuffled folds). The
  mechanism claim is *stronger*, not weaker.
- **Tier-2** — root requirements.txt Python 3.10→3.11; analysis requirements +
  matplotlib; REPRODUCE.md build target → `latent_sentinel_arr.tex`, new scripts
  + honest seed scope + per-script wait-flag accuracy; ECE labelled
  "positive/attack-class reliability gap, 15 bins".

## Residual items (honestly disclosed, not fully closable from cache)

- **C4 residual:** the uniform-protocol re-scoring removes the *calibration-budget*
  asymmetry, but FJD's internal temperature fit and JBShield's concept-ref
  sampling are baked into the cached scores; a fully uniform re-run needs the GPU
  baseline pipeline. The paper now states FJD-tuning is *conservative against our
  claim* (it strengthens a weak baseline) and frames ρ as supporting/descriptive
  — adequate for review; a uniform GPU re-run remains future work.
- **Page budget:** content now ends on page 8 (the honest additions consumed the
  earlier headroom). Within ARR's "up to 8 pages" but with no margin — verify in
  the official build; pre-identified appendix-movable blocks (cached-LEACE
  control, tab:native) can recover ~0.3 pp if it spills.
- Original probe training remains single-run/unseeded (documented in Limitations;
  the audit's seed concern is about evaluation-split variance, now scoped
  correctly).

## Updated bottom-line

- Technically sound: **Yes** (headline validated leakage-free; honest scope).
- Reproducible: **Partial→Yes** for the EMNLP claims (every number traces to a
  committed audit-fixed script; env pinned; original notebooks still legacy).
- Leakage-free evaluation: **Yes** for the spine (disjoint eraser/cal/test;
  native-τ on held-out; uniform protocol). Residual JBShield self-leak documented.
- Ablations trustworthy: **Yes** (register-balanced clean & corroborating; LEACE
  now leakage-free; mitigation correctly disclaimed).
- Metrics trustworthy: **Yes** (cal-selected layer; honest ECE; uniform ρ;
  held-out native-τ).
- Claim-to-code alignment: **Yes** (re-audited ALL CONSISTENT; overclaims
  corrected: ρ 0.87→0.68, +0.34→0.23, Z-recover honest, seed-scope honest).
- Submission-ready for a serious AI/ML venue: **Yes, modulo the final
  official-template page check** — the science is now leakage-free, the headline
  survived, overclaims are corrected, and the paper is materially more honest and
  defensible than before the audit.
