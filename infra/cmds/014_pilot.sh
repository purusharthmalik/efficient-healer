# Pilot: first 20 train-pool tasks x 2 rollouts with the base student.
# Written into the real pool file, so the full collection run resumes from it.
set -u
cd /work
export PYTHONPATH=/work/sync/code
bash /work/sync/code/cmds/serve_student.sh 0.15 || exit 1
.venv-tune/bin/python -m selfheal.rollout --tasks out/splits.json --split pool --limit 20 --k 2 \
  --out out/pool/rollouts.jsonl --tag base-pool --concurrency 32 2>&1 | grep -E "^\[rollout" 
.venv-tune/bin/python -m selfheal.summary out/pool/rollouts.jsonl --json out/pool/pilot_summary.json
.venv-tune/bin/python -m selfheal.buckets out/pool/rollouts.jsonl --out out/pool/pilot_buckets.json
.venv-tune/bin/python -m selfheal.failures out/pool/rollouts.jsonl --out out/pool/pilot_failures.json 2>&1 | grep -v Warning
