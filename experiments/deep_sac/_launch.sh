#!/bin/bash
cd /home/taranjeet.bakshi/code_search_harness
source .venv/bin/activate 2>/dev/null || true
set -a; source phase4/.secrets 2>/dev/null; source ~/taxonomy/.env 2>/dev/null; set +a
exec python -m experiments.deep_sac.run_deep_sac 50 4
