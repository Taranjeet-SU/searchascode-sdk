#!/usr/bin/env bash
set -u
cd /home/taranjeet.bakshi/code_search_harness
source .venv/bin/activate 2>/dev/null || source phase1/.venv/bin/activate 2>/dev/null
export $(grep -v '^#' ~/taxonomy/.env 2>/dev/null | xargs) 2>/dev/null
echo "===== $(date +%H:%M:%S) HOTPOT LEARN: mine rules =====" | tee -a phase2/runs/hotpot_learn.log
python -m phase2.learn_rules --dataset hotpotqa --n 200 --max-cases 40 >> phase2/runs/hotpot_learn.log 2>&1
echo "===== $(date +%H:%M:%S) HOTPOT LEARN: calibrate judge =====" | tee -a phase2/runs/hotpot_learn.log
python -m phase2.align_prompts --dataset hotpotqa --n 60 >> phase2/runs/hotpot_learn.log 2>&1
echo "===== $(date +%H:%M:%S) HOTPOT LEARN COMPLETE =====" | tee -a phase2/runs/hotpot_learn.log
tail -20 phase2/runs/hotpot_learn.log
