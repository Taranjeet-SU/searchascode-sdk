#!/usr/bin/env bash
set -u
cd /home/taranjeet.bakshi/code_search_harness
source .venv/bin/activate 2>/dev/null || source phase1/.venv/bin/activate 2>/dev/null
set -a; source phase4/.secrets; source ~/taxonomy/.env 2>/dev/null; set +a
python -m phase4.altera_eval_agent --n "${1:-195}" --k "${2:-6}" --workers "${3:-8}" --max-hops "${4:-3}"
