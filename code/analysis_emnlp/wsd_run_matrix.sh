#!/bin/bash
set -u
PY="${PYTHON:-python}"
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
B=code/analysis_emnlp/wsd_baselines.py; L=data/emnlp2026
echo "[matrix] phase1: FJD/llama (gpu1) + JBShield/llama (gpu2)"
$PY -u $B --method fjd      --model llama_aligned --gpu 1 --batch-size 16 > $L/wsd_fjd_llama.log 2>&1 &
P1=$!
$PY -u $B --method jbshield --model llama_aligned --gpu 2 --batch-size 16 > $L/wsd_jbs_llama.log 2>&1 &
P2=$!
wait $P1 $P2
echo "[matrix] phase1 done. phase2: GradSafe/qwen (gpu1) + GradSafe/llama (gpu2)"
$PY -u $B --method gradsafe --model qwen_aligned  --gpu 1 > $L/wsd_gs_qwen.log  2>&1 &
P3=$!
$PY -u $B --method gradsafe --model llama_aligned --gpu 2 > $L/wsd_gs_llama.log 2>&1 &
P4=$!
wait $P3 $P4
echo "[matrix] WSD_MATRIX_DONE"
