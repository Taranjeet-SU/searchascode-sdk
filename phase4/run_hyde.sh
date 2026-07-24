#!/usr/bin/env bash
set -u
cd /home/taranjeet.bakshi/code_search_harness
source .venv/bin/activate 2>/dev/null || source phase1/.venv/bin/activate 2>/dev/null
set -a; source phase4/.secrets; source ~/taxonomy/.env 2>/dev/null; set +a
python -m phase4.altera_hyde --n 195 --k 8 --workers 3
