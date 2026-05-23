# REPRODUCE.md — Latent Sentinel (EMNLP 2026 artifact)

Environment: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124`
(Python 3.11, CUDA 12.4; see file header). Interpreter used for all results:
`<conda-env>/bin/python` (substitute your venv).

Splits are seeded (`seed=42` for probe/in-dist splits unless varied; the disjoint
benign 3-way split uses `seed=20260520`). The "3 seeds" (41/42/43) resample the
*evaluation split* of a single inference run — probes are single-run (original
training is unseeded; see CHECK.md H-5 — re-train sweep recommended). GPU scripts
auto-pick the freest device; `pivot_register_erase_remedy.py` /
`pivot_mitigation_adaptive.py` accept `--wait-vram <MiB>`,
`wsb_benign_stress_infer.py` uses `--wait-for-vram`, the `wsc_*`/`wsd_*` scripts
take only `--gpu`.

## Tables and figures → producer scripts

| Paper artifact | Script (in `code/analysis_emnlp/`) | Output |
|---|---|---|
| Table 1 (`tab:wsa`) — Calibration-selected balanced metrics | `ws_a_balanced_metrics.py --all --seed 42` (also seeds 41/43 for ±std) | `data/emnlp2026/ws_a/*__balanced.json`, `CONSOLIDATED_corrected_metrics.csv` |
| Table 2 (`tab:ood`) — Benign-stress FPR at conformal τ | `wsb_benign_stress_infer.py --models qwen_aligned,llama_aligned --gpu N` | `data/emnlp2026/wsb/benign_stress_fpr.json` |
| §5 No-model surface classifier | `wsb_register_confound.py --all-best` | `data/emnlp2026/wsb/register_confound*.json` |
| Table 3 (`tab:native`) — native 95%-TPR threshold | `wsf_native_threshold.py` | `data/emnlp2026/wsf_native_threshold.json` |
| Table 5 (`tab:baselines`) — 6 detectors uniform protocol | `wsf_tradeoff_uniform.py`, `wsd_baselines.py`, `wsd_hiddendetect_baseline.py` | `data/emnlp2026/wsf_tradeoff_uniform.json`, `data/emnlp2026/wsd/*` |
| Bootstrap CIs in Table 5 | `wsf_bootstrap_ci.py` (audit fix M-2: paired CI) | `data/emnlp2026/wsf_bootstrap_ci.json` |
| **Tables 4 & 6 (`tab:erase`, `tab:deploy`) — register erasure spine + recipe** | `pivot_heldout_remedy.py` (audit fix B1: leakage-free disjoint 3-way split; CHECK fix H-1: `Hin[cal_b]`-only LEACE fit) | `data/emnlp2026/pivot/heldout_remedy.json` |
| Bootstrap CIs in Tables 4, 6 | `pivot_bootstrap_tables.py` (CHECK fix H-2: per-bootstrap rng seed) | `data/emnlp2026/pivot/bootstrap_main_tables.json` |
| §Erase universal alignment Procrustes ≤0.04 (Fig 3) | `pivot_subspace_alignment.py` (CHECK fix H-3: random-basis null distribution; report p-value) | `data/emnlp2026/pivot/subspace_alignment.json` |
| §Adaptive Evasion (embedding upper bound) | `wsc_adaptive_probe_evasion.py --model {qwen,llama}_aligned --gpu N --seed 42` | `data/emnlp2026/wsc/adaptive_evasion.json` |
| §Adaptive Evasion (discrete GCG; CHECK fix C-3/H-4: unified suffix position, `add_special_tokens=True`) | `wsc_gcg.py --model {qwen,llama}_aligned --gpu N --seed 42` | `data/emnlp2026/wsc/gcg_evasion.json` |
| §fix B1' Adaptive margin (recipe vs baseline; CHECK fix C-4: closed-form head equality assertion) | `pivot_adaptive_margin_recipe.py` | `data/emnlp2026/pivot/adaptive_margin_recipe.json` |
| §fix Adaptive scope (recipe vs cached GCG suffix) | `pivot_adaptive_on_recipe.py --gpu N` | `data/emnlp2026/pivot/adaptive_on_recipe.json` |
| §fix Sensitivity / E3 Wilcoxon / E4 latency / E6 split-seed (Table 8 `tab:seedsens`) | `pivot_sensitivity.py` (CHECK fix H-2: per-seed in-dist rng) | `data/emnlp2026/pivot/sensitivity.json` |
| §Limitations LOSO Cross-source (Table 7 `tab:loso`) | `pivot_loso_crosssource.py` | `data/emnlp2026/pivot/loso_crosssource.json` |
| §App. B Qualitative examples | `pivot_qualitative_examples.py` | `data/emnlp2026/pivot/qualitative_examples.json` |
| §App. A rank-3 interpretation (question-mark correlations) | `pivot_subspace_viz.py` (also produces Fig 2 `fig_rank3_register.pdf`) | `data/emnlp2026/pivot/rank3_interpretation.json` + `paper/fig_rank3_register.pdf` |
| Figure 1 — In-dist AUC vs OOD FPR scatter (`fig_tradeoff.pdf`) | `make_tradeoff_figure.py` (reads `wsf_tradeoff_uniform.json`) | `paper/fig_tradeoff.pdf` |
| Figure 2 — rank-3 PCA scatter (`fig_rank3_register.pdf`) | `pivot_subspace_viz.py` | `paper/fig_rank3_register.pdf` |
| Figure 3 — Procrustes overlay (`fig_subspace_overlay.pdf`) | `pivot_appendix_figures.py` (reads `subspace_alignment.json`) | `paper/fig_subspace_overlay.pdf` |
| Figure 4 — Operating-point scatter (`fig_operating_point.pdf`) | `pivot_appendix_figures.py` (reads `deployable_operating_point.json`) | `paper/fig_operating_point.pdf` |
| Figure 5 — Adaptive δ_min bar chart (`fig_adaptive_margin.pdf`) | `pivot_appendix_figures.py` (reads `adaptive_margin_recipe.json`) | `paper/fig_adaptive_margin.pdf` |
| Per-sample probe scores (`tab:wsa` and §5 input) | `recompute_per_sample_scores.py` (CHECK fix B2: CLI replacement for legacy notebooks) | `data/predictions/stander_jailbreak_eval*/*.csv`, `data/detailed_evaluation_results.pkl` |
| Pivot hidden-state cache prerequisite (required by all `pivot_*` scripts) | `pivot_register_erase_remedy.py --model {qwen,llama,mistral}_aligned --gpu N` (CHECK fix: this is the **unique producer of `hs_*.npz`**; not "superseded" — kept as cache regenerator). The `register_erase_remedy.json` it emits is itself superseded by `heldout_remedy.json`. | `data/emnlp2026/pivot/hs_qwen_aligned_L20.npz`, `hs_llama_aligned_L16.npz`, `hs_mistral_aligned_L16.npz` |
| (deprecated; do not cite) | `pivot_deployable_operating_point.py` (leaked original; superseded by `heldout_remedy`) | `data/emnlp2026/pivot/deployable_operating_point.json` |

## Determinism

`code/analysis_emnlp/_setup.py` sets `torch.manual_seed`, `numpy.random.seed`,
`random.seed`, `torch.use_deterministic_algorithms(True)`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and disables cudnn benchmark. Import it at
the top of every CLI script.

## Critical environment facts (see repo `CLAUDE.md`)

- Backbones for forward-pass: only the 3 **aligned** models are recoverable.
  `qwen_aligned` = HF `Qwen/Qwen2.5-7B-Instruct`; `llama_aligned` =
  `NousResearch/Meta-Llama-3.1-8B-Instruct` (ungated mirror; `meta-llama/*` is
  gated); `mistral_aligned` = `unsloth/mistral-7b-instruct-v0.3` (ungated mirror;
  `mistralai/Mistral-7B-Instruct-v0.3` is gated).
- **Both `*_JB` alignment-degraded checkpoints are incomplete in this repo**
  (corrupt zip / 2-of-7 shards) → unrecoverable. Their calibration is covered by
  the WS-A re-analysis of on-disk per-sample scores in `data/predictions/`.
- Probe recipe (matches `code/probe_training/jailbreak_layer_15.py`): truncate to L
  blocks, `nn.Linear(hidden,2)` head from `data/trained_probes/stander_jailbreak/`,
  last-token logits, P(jailbreak) = softmax[:,0]. **CHECK fix C-1**: training and
  inference must use the same padding-side; re-train with `padding_side="left"` +
  `set_seed(42)` (see CHECK.md G6).
- HuggingFace gating: `lmsys/toxic-chat` requires `HF_TOKEN` + license accept.
  Set `export HF_TOKEN=...` before `wsb_fetch_benign_stress.py`.

## Best layers (WS-A) used downstream

Qwen2.5-7B aligned L20 · Qwen2.5-7B_JB L16 · Llama-3.1-8B aligned L16 ·
Llama-3.1-8B_JB L15 (re-analysis only) · Mistral-7B aligned L16 (mid-depth, not
WS-A-tuned; per-layer profile in App. — see CHECK fix M-5).

## Manuscript build

`cd paper && pdflatex -interaction=nonstopmode latent_sentinel_arr.tex` (twice
for refs). Single source of truth; camera-ready transformation in the header
comment.

## Notes / honest scope

- The aggregation-vs-adaptive mitigation script `pivot_mitigation_adaptive.py` is
  retained for transparency but its result is **methodologically invalid** (a
  single-probe GCG suffix does not transfer to a re-scored aggregated detector);
  the paper makes **no aggregation-robustness claim** and flags a faithful
  adaptive evaluation of an aggregated detector as future work.
- Alignment-degraded backbones are characterized via on-disk per-sample scores
  only; their checkpoints are unrecoverable.

## Audit fixes applied (see CHECK.md)

- **C-1**: padding-side fix in probe training (re-train recommended)
- **C-2**: `pivot_register_erase_remedy.py` output renamed `*.SUPERSEDED.json`
- **C-3/H-4**: unified chat-template/suffix position between `wsc_*` scripts
- **C-4**: closed-form head equality assertion in adaptive-margin scripts
- **H-1**: LEACE fits use `Hin[cal_b]` only (in-dist test partition excluded)
- **H-2**: `pivot_sensitivity.py` and `pivot_bootstrap_tables.py` use per-seed in-dist rng
- **H-3**: Procrustes null distribution in `pivot_subspace_alignment.py`
- **H-6**: post-LEACE register-confound check (new `pivot_postleace_confound.py`)
- **H-7**: native-protocol companion table for baselines (`wsf_native_protocol_companion.py`)
- **H-9**: JBShield fit on `cal` indices only
- **B2**: `recompute_per_sample_scores.py` CLI replacement for legacy notebooks
- **M-9**: `_setup.py` enforces CUDA determinism across all scripts
