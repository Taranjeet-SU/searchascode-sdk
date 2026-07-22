#!/usr/bin/env bash
# Stage 2 (re-run): trec-covid (zip already valid) now; scidocs once its download validates.
set -u
cd /home/taranjeet.bakshi/code_search_harness
source .venv/bin/activate 2>/dev/null || source phase1/.venv/bin/activate 2>/dev/null
export $(grep -v '^#' ~/taxonomy/.env 2>/dev/null | xargs) 2>/dev/null

run_when_valid () {
  local d="$1"
  # wait until the zip passes testzip (download finished) — cap ~20 min
  for _ in $(seq 1 80); do
    if python -c "import zipfile;zipfile.ZipFile('phase2/data/$d.zip').testzip()" 2>/dev/null; then break; fi
    sleep 15
  done
  if ! python -c "import zipfile;zipfile.ZipFile('phase2/data/$d.zip').testzip()" 2>/dev/null; then
    echo "===== $d zip STILL BAD after wait, skipping =====" | tee -a phase2/runs/campaign.log; return
  fi
  echo "===== $(date +%H:%M:%S) START $d (stage2) =====" | tee -a phase2/runs/campaign.log
  python -m phase2.beir_run --dataset "$d" --ingest --n 40 >> phase2/runs/${d}.log 2>&1
  echo "===== $(date +%H:%M:%S) DONE $d rc=$? =====" | tee -a phase2/runs/campaign.log
  tail -9 phase2/runs/${d}.log | tee -a phase2/runs/campaign.log
}

run_when_valid trec-covid
run_when_valid scidocs
echo "===== $(date +%H:%M:%S) STAGE2 (re-run) COMPLETE =====" | tee -a phase2/runs/campaign.log
