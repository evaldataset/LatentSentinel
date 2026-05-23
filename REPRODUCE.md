# REPRODUCE.md — Latent Sentinel (EMNLP 2026 artifact)

## 30-minute reviewer quickstart (CPU only, no model download)

Verifies the headline numbers from cached states already in the bundle.

```bash
pip install -r requirements.txt
python code/analysis_emnlp/pivot_heldout_remedy.py           # Tables 4 + 6 (recipe vs baseline)
python code/analysis_emnlp/pivot_subspace_alignment.py       # rank-3 Procrustes (Fig 3)
python code/analysis_emnlp/pivot_subspace_alignment_null.py  # Procrustes null p-values (App. G)
python code/analysis_emnlp/pivot_postleace_confound.py       # App. H residual-confound test
python code/analysis_emnlp/wsf_tradeoff_3backbone.py         # 3-backbone Spearman rho (Fig 1 ext.)
python code/analysis_emnlp/wsf_native_threshold.py           # Native-threshold sanity
diff data/emnlp2026/pivot/heldout_remedy.json /tmp/  # byte-identical re-derivation
```

All 4 producers emit JSONs byte-identical to the on-disk versions (verified
2026-05-23). GPU-only re-extraction of `hs_*.npz` caches is **not** required
for re-deriving Tables 4-6, App. G/H, or Fig 1/3.

## Producer ↔ artifact map (full)

Environment: `pip install -r requirements.txt` (Python 3.11, CUDA 12.4; see header).
Interpreter used for all results:
`<conda-env>/bin/python` (substitute your venv).
Splits are seeded (`seed=42` for probe/in-dist splits; the disjoint benign
3-way split uses a fixed seed). The "3 seeds" (41/42/43) resample the
\emph{evaluation split} of a single inference run — probes are single-run
(original training is unseeded). GPU scripts auto-pick the freest device;
`pivot_register_erase_remedy.py` / `pivot_mitigation_adaptive.py` accept
`--wait-vram <MiB>`, `wsb_benign_stress_infer.py` uses `--wait-for-vram`, the
`wsc_*`/`wsd_*` scripts take only `--gpu`.

| Paper artifact | Script (in `code/analysis_emnlp/`) | Output |
|---|---|---|
| §Calibration Collapse, ECE 0.20–0.33, seed study | `ws_a_balanced_metrics.py` | `data/emnlp2026/ws_a/*__balanced.json` |
| Benign-stress set (XSTest/OR-Bench/ToxicChat) | `wsb_fetch_benign_stress.py` | `data/emnlp2026/wsb/benign_stress.jsonl` |
| §Over-Refusal Under Distribution Shift | `wsb_benign_stress_infer.py --model {qwen,llama}_aligned` | `data/emnlp2026/wsb/*` |
| §The Signal Is Benign Register (no-model control) | `wsb_register_confound.py` | `data/emnlp2026/wsb/register_confound*.json` |
| §Adaptive Evasion (embedding + discrete GCG) | `wsc_gcg.py`, `wsc_adaptive_probe_evasion.py` | `data/emnlp2026/wsc/*` |
| Table (baselines): 6 detectors × 2 models | `wsd_baselines.py`, `wsd_hiddendetect_baseline.py` | `data/emnlp2026/wsd/baselines.json` |
| §Over-Refusal native-threshold control (Table tab:native) | `wsf_native_threshold.py` (audit-fixed C5: tau on CAL jb, sign-oriented) | `data/emnlp2026/wsf_native_threshold.json` |
| **Trade-off ρ + Table tab:baselines (uniform protocol)** | `wsf_tradeoff_uniform.py` (audit fix C4/M1; emits the 12-cell ρ, per-model, LOO) | `data/emnlp2026/wsf_tradeoff_uniform.json` |
| **§Register Erasure spine + §Recipe (Tables tab:erase, tab:deploy)** | `pivot_heldout_remedy.py` (audit fix **B1**: source-stratified disjoint eraser/cal/test; leakage-free) | `data/emnlp2026/pivot/heldout_remedy.json` |
| **§fix Adaptive scope (recipe vs cached GCG suffix)** | `pivot_adaptive_on_recipe.py --gpu N` (CHECK fix Critical-1; transfers cached suffix to the recipe; needs GPU) | `data/emnlp2026/pivot/adaptive_on_recipe.json` |
| **App. J Recipe-aware white-box adaptive attack** (Table tab:recipe-aware) | `wsc_adaptive_recipe_aware.py --model {qwen,llama}_aligned --gpu N` (attacker knows LEACE eraser; re-optimizes 20-token embedding suffix against erased probe; ~25 min/backbone on A100) | `data/emnlp2026/wsc/adaptive_recipe_aware.json` |
| **§fix Sensitivity / E3 Wilcoxon / E4 latency / E6 split-seed / E7 LEACE rank** | `pivot_sensitivity.py` (CHECK fixes E3/E4/E6/E7; CPU only) | `data/emnlp2026/pivot/sensitivity.json` |
| **§Limitations Cross-source generalization (LOSO)** | `pivot_loso_crosssource.py` (CHECK fix Critical-2: leave-one-source-out from cached states; CPU only) | `data/emnlp2026/pivot/loso_crosssource.json` |
| Hero scatter `fig_tradeoff.pdf` | `make_tradeoff_figure.py` (audit fix M2: reads `wsf_tradeoff_uniform.json`, no hardcoded points) | `paper/fig_tradeoff.pdf` |
| (superseded, kept for the audit trail) | `pivot_register_erase_remedy.py`, `pivot_deployable_operating_point.py` (the leaked originals; see `AUDIT.md` B1) | `register_erase_remedy.json`, `deployable_operating_point.json` |

Manuscript build: `cd paper && latexmk -pdf latent_sentinel_arr.tex`
(this is the ARR submission; `latent_sentinel_emnlp2026_anon.tex` is an identical
copy, `latent_sentinel_emnlp2026.tex` is the [final]+named camera-ready precursor).
All numeric claims trace to the audit-fixed scripts above; see `AUDIT.md`.

Notes / honest scope:
- The aggregation-vs-adaptive mitigation script `pivot_mitigation_adaptive.py` is
  retained for transparency but its result is **methodologically invalid** (a
  single-probe GCG suffix does not transfer to a re-scored aggregated detector); the
  paper makes **no aggregation-robustness claim** and flags a faithful adaptive
  evaluation of an aggregated detector as future work.
- Alignment-degraded backbones are characterized via on-disk per-sample scores only;
  their checkpoints are unrecoverable (corrupt archive / partial shard set) and are
  not re-inferred. All forward-pass experiments use aligned backbones.
