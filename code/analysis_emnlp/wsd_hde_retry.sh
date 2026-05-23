#!/bin/bash
set -u
PY="${PYTHON:-python}"
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
L=data/emnlp2026; J=data/emnlp2026/wsd/baselines.json
have(){ $PY -c "import json,sys;d=json.load(open('$J'));sys.exit(0 if '$1' in d.get('hiddendetect_exact',{}) else 1)" 2>/dev/null; }
for m in qwen_aligned llama_aligned; do
  for try in 1 2 3 4 5 6 7 8; do
    have $m && { echo "[hde-retry] $m already done"; break; }
    G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits|sort -t, -k2 -nr|head -1)
    gi=$(echo $G|cut -d, -f1); gf=$(echo $G|cut -d, -f2|tr -d ' ')
    if [ "$gf" -lt 18000 ]; then echo "[hde-retry] $m try$try: max free ${gf}MiB<18000, wait 90s"; sleep 90; continue; fi
    echo "[hde-retry] $m try$try on GPU$gi (${gf}MiB free)"
    $PY -u code/analysis_emnlp/wsd_baselines.py --method hiddendetect_exact --model $m --gpu $gi --batch-size 16 > $L/wsd_hde_${m}.log 2>&1
    have $m && { echo "[hde-retry] $m OK"; break; }
    echo "[hde-retry] $m try$try failed (race/err); sleep 60"; sleep 60
  done
done
echo "HDE_RETRY_DONE"
