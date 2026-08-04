#!/usr/bin/env bash
set -u
cd /home/taranjeet.bakshi/code_search_harness
source .venv/bin/activate 2>/dev/null || source phase1/.venv/bin/activate 2>/dev/null
export $(grep -v '^#' ~/taxonomy/.env 2>/dev/null | xargs) 2>/dev/null
for d in scifact nfcorpus scidocs arguana; do
  echo "===== $(date +%H:%M:%S) LEARN $d =====" | tee -a phase2/runs/learn_sweep.log
  python -m phase2.learn_rules --dataset "$d" --n 150 --max-cases 30 >> phase2/runs/learn_sweep.log 2>&1
  echo "===== $(date +%H:%M:%S) IMPACT $d =====" | tee -a phase2/runs/learn_sweep.log
  python -m phase2.impact_eval --dataset "$d" --n 150 >> phase2/runs/learn_sweep.log 2>&1
  grep -E "learned-profile impact|dense \(raw\)|synonym-expand" phase2/runs/learn_sweep.log | tail -3 | tee -a phase2/runs/learn_sweep.log
done
echo "===== $(date +%H:%M:%S) LEARN SWEEP COMPLETE =====" | tee -a phase2/runs/learn_sweep.log
