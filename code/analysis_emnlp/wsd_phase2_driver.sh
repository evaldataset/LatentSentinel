#!/bin/bash
set -u
PY="${PYTHON:-python}"
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
L=data/emnlp2026
need=17000
pick(){ nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits|sort -t, -k2 -nr|head -1; }
waitgpu(){ while :; do read g f < <(pick|tr ',' ' '); [ "$f" -ge $need ] && { echo $g; return; }; sleep 120; done; }
echo "[p2] waiting for >=${need}MiB free GPU..."
G=$(waitgpu); echo "[p2] using GPU$G for hiddendetect_exact+gcg lane"
for m in qwen_aligned llama_aligned; do
  $PY -u code/analysis_emnlp/wsd_baselines.py --method hiddendetect_exact --model $m --gpu $G --batch-size 16 > $L/wsd_hde_$m.log 2>&1
done
for m in qwen_aligned llama_aligned; do
  G=$(waitgpu)
  $PY -u code/analysis_emnlp/wsc_gcg.py --model $m --gpu $G > $L/wsc_gcg_$m.log 2>&1
done
echo "[p2] WSD_PHASE2_DONE"
