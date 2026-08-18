#!/usr/bin/env bash
# Serial 5-dataset base-numbers campaign (one at a time: shared GPU).
set -u
cd /home/taranjeet.bakshi/code_search_harness
source .venv/bin/activate 2>/dev/null || source phase1/.venv/bin/activate 2>/dev/null
export $(grep -v '^#' ~/taxonomy/.env 2>/dev/null | xargs) 2>/dev/null
mkdir -p phase2/runs
N=40
for d in scifact arguana scidocs trec-covid nfcorpus; do
  echo "===== $(date +%H:%M:%S) START $d =====" | tee -a phase2/runs/campaign.log
  python -m phase2.beir_run --dataset "$d" --ingest --n $N >> phase2/runs/${d}.log 2>&1
  echo "===== $(date +%H:%M:%S) DONE $d rc=$? =====" | tee -a phase2/runs/campaign.log
  tail -8 phase2/runs/${d}.log | tee -a phase2/runs/campaign.log
done
echo "===== $(date +%H:%M:%S) CAMPAIGN COMPLETE =====" | tee -a phase2/runs/campaign.log
