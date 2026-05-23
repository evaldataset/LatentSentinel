# Latent Sentinel — Anonymous Code Bundle (ARR May-25 / EMNLP 2026)

Anonymous artifact for the submission **"Linearly Readable, Not Robust: How Linear Jailbreak Probes Conflate Benign Register with Adversarial Intent"** (under double-blind review at ACL Rolling Review May-2025 cycle, target venue EMNLP 2026).

This bundle is mirrored at [anonymous.4open.science](https://anonymous.4open.science) for double-blind review; the underlying GitHub repository is hosted under an eval-only organisation account that contains no author-identifying metadata.

## What this contains

```
code/analysis_emnlp/          # All experiment / analysis scripts (audit-fixed)
  ├── pivot_heldout_remedy.py         # Tables 4 & 6 producer (LEACE + conformal recipe)
  ├── pivot_subspace_alignment.py     # Rank-3 register subspace + Procrustes
  ├── pivot_subspace_alignment_null.py# Procrustes null distribution (App. F)
  ├── pivot_postleace_confound.py     # Post-LEACE register-confound (App. G)
  ├── pivot_loso_crosssource.py       # LOSO cross-source (App. D)
  ├── pivot_loso_orbench_include.py   # OR-Bench-H included LOSO (App. D')
  ├── pivot_mistral_layer_profile.py  # Mistral per-layer (App. H)
  ├── pivot_appendix_figures.py       # Figures 3-5 producer
  ├── pivot_adaptive_margin_recipe.py # Closed-form δ_min (App. C)
  ├── pivot_adaptive_on_recipe.py     # Recipe vs cached GCG suffix
  ├── pivot_sensitivity.py            # Split-seed sensitivity (App. E)
  ├── pivot_bootstrap_tables.py       # Bootstrap CIs for headline tables
  ├── pivot_qualitative_examples.py   # Qualitative examples (App. B)
  ├── pivot_register_erase_remedy.py  # hs_*.npz cache producer (run first)
  ├── wsd_arditi_baseline.py          # Arditi 2024 refusal-direction baseline
  ├── wsd_baselines.py                # HiddenDetect / FJD / GradSafe / JBShield baselines
  ├── wsd_hiddendetect_baseline.py    # HiddenDetect (exact reimplementation)
  ├── wsc_adaptive_probe_evasion.py   # Universal embedding-suffix attack
  ├── wsc_gcg.py                      # Universal discrete-GCG attack
  ├── wsc_gcg_mistral.py              # Mistral GCG (audit-fix probe)
  ├── wsc_adaptive_mistral.py         # Mistral embedding attack
  ├── wsc_aggregator_transfer.py      # 3-layer majority-vote (App. J)
  ├── ws_a_balanced_metrics.py        # Calibration audit (Table 1)
  ├── wsb_register_confound.py        # No-model surface classifier (§5)
  ├── wsb_benign_stress_infer.py      # OOD over-refusal (Table 2)
  ├── wsb_fetch_benign_stress.py      # Benign-stress prep
  ├── wsf_tradeoff_uniform.py         # Uniform protocol Spearman ρ (12-cell)
  ├── wsf_tradeoff_3backbone.py       # 3-backbone Spearman ρ (20-cell)
  ├── wsf_native_threshold.py         # Native 95%-TPR threshold (Table 3)
  ├── wsf_bootstrap_ci.py             # Bootstrap CI utilities
  ├── recompute_per_sample_scores.py  # B2 CLI (legacy notebook replacement)
  ├── make_tradeoff_figure.py         # Figure 1 producer
  ├── _setup.py                       # CUDA determinism + seed utility
  ├── REPRODUCE.md                    # 25-row producer ↔ table/figure map
  └── requirements.txt                # Pinned environment (Python 3.11 + CUDA 12.4)

code/probe_training/
  └── retrain_probe_audit_fixed.py    # C-1 + H-5 audit-fix probe retraining

data/emnlp2026/
  ├── pivot/*.json                    # All headline-table source data
  ├── wsb/{benign_stress.jsonl,*.json}
  ├── wsc/*.json                      # Adaptive evasion results
  ├── wsd/*.json + *.npz              # Baseline scores
  ├── ws_a/*.json + *.csv             # Calibration metrics
  └── wsf_*.json                      # Trade-off + bootstrap CI artifacts

data/trained_probes_fixed/
  ├── qwen_aligned/                   # 40 probes (L15-22 × 5 seeds) + summary
  ├── llama_aligned/                  # 35 probes (L12-18 × 5 seeds) + summary
  └── mistral_aligned/                # 35 probes (L12-18 × 5 seeds) + summary

REPRODUCE.md            # Top-level entry point: producer-script ↔ table/figure map
AUDIT.md                # Method-level audit log
CHECK.md                # Pre-submission critical audit (1629 lines, A-S sections)
requirements.txt        # Environment pin
_ANON_VERIFY.txt        # Identifier-leak verification stamp
```

## Quickstart

```bash
# 1. Install pinned environment (Python 3.11 + CUDA 12.4 + 80GB-class GPU recommended)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# 2. CPU-only re-analysis from cached states (no GPU required after caches exist)
python code/analysis_emnlp/pivot_heldout_remedy.py            # Tables 4 + 6
python code/analysis_emnlp/pivot_subspace_alignment.py        # Procrustes
python code/analysis_emnlp/pivot_subspace_alignment_null.py   # Procrustes null
python code/analysis_emnlp/pivot_postleace_confound.py        # Register confound
python code/analysis_emnlp/pivot_loso_crosssource.py          # LOSO
python code/analysis_emnlp/pivot_sensitivity.py               # Split-seed sensitivity
python code/analysis_emnlp/wsf_tradeoff_3backbone.py          # 3-backbone ρ

# 3. GPU experiments (forward-pass through backbones)
#    Producer of hs_*.npz pivot caches (required by all pivot_* scripts above)
python code/analysis_emnlp/pivot_register_erase_remedy.py \
    --model qwen_aligned --gpu 0 --wait-vram 17

#    Baselines / adaptive attacks
python code/analysis_emnlp/wsd_baselines.py --method fjd --model qwen_aligned --gpu 0
python code/analysis_emnlp/wsc_adaptive_probe_evasion.py --model qwen_aligned --gpu 0
python code/analysis_emnlp/wsc_gcg.py --model qwen_aligned --gpu 0
```

Full producer ↔ artifact map: see [`REPRODUCE.md`](REPRODUCE.md) (25 rows covering all paper tables / figures).

## Reproducibility caveats

- **HuggingFace gating**: `lmsys/toxic-chat` requires `HF_TOKEN` + license accept; the backbones use ungated mirrors (`Qwen/Qwen2.5-7B-Instruct`, `NousResearch/Meta-Llama-3.1-8B-Instruct`, `unsloth/mistral-7b-instruct-v0.3`).
- **`hs_*.npz` pivot caches** (~3 GB total) are excluded from this bundle (size). Regenerate by running `pivot_register_erase_remedy.py` once per backbone (each ~15 minutes on an 80 GB-class GPU). All downstream pivot_* analyses are then CPU-only.
- **Probe weights**: audit-fix retrained probes (`data/trained_probes_fixed/`) are included (each ~5–15 MB).
- **Determinism**: `_setup.py` sets `torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and cudnn deterministic mode. Import it at the top of each CLI.

## License

Apache-2.0 (see [`LICENSE`](LICENSE)).
