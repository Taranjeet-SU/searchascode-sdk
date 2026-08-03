#!/usr/bin/env bash
set -u
cd /home/taranjeet.bakshi/code_search_harness
source .venv/bin/activate 2>/dev/null || source phase1/.venv/bin/activate 2>/dev/null
set -a; source ~/taxonomy/.env 2>/dev/null; set +a
# args: target(1000) workers(8) n_docs(2)
python -m experiments.multi_hop_synth_queries.generate "${1:-1000}" "${2:-8}" "${3:-2}" 2>&1
