#!/bin/bash
# WS-F: WS-C adaptive-evasion across seeds for CIs (seed 42 already in main run).
PY="${PYTHON:-python}"
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
for seed in 41 43; do
  for m in qwen_aligned llama_aligned; do
    echo "=== WS-C $m seed $seed ==="
    $PY -u code/analysis_emnlp/wsc_adaptive_probe_evasion.py --model $m --gpu 1 \
        --seed $seed 2>&1 | grep -E 'evasion rate|recall @ conformal|step  199' \
        | sed "s/^/[$m s$seed] /"
    # stash per-seed result
    $PY -c "import json,shutil,os; f='data/emnlp2026/wsc/adaptive_evasion.json'; \
            d=json.load(open(f)); \
            json.dump(d.get('$m',{}), open(f'data/emnlp2026/wsc/seed${seed}_$m.json','w'))"
  done
done
echo "WSF_WSC_DONE"
