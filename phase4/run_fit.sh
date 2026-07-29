#!/usr/bin/env bash
set -u
cd /home/taranjeet.bakshi/code_search_harness
source .venv/bin/activate 2>/dev/null || source phase1/.venv/bin/activate 2>/dev/null
set -a; source phase4/.secrets 2>/dev/null; source ~/taxonomy/.env 2>/dev/null; set +a
export ALTERA_OS=http://localhost:8056
python -m phase4.altera_fit "${1:-5000}" "${2:-2}" 2>&1
