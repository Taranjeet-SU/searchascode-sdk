#!/usr/bin/env bash
set -u
cd /home/taranjeet.bakshi/code_search_harness
source .venv/bin/activate 2>/dev/null || source phase1/.venv/bin/activate 2>/dev/null
set -a; source ~/taxonomy/.env 2>/dev/null; set +a
python -m experiments.multi_hop_synth_queries.eval_recall "${1:-150}" "${2:-6}" 2>&1
