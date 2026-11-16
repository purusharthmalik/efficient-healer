# Round 1 collection: base Tiny Aya on the 240 train-pool tasks, k=4, with the stop rule.
set -u
cd /work
export PYTHONPATH=/work/sync/code
bash /work/sync/code/cmds/serve_student.sh 0.15 || exit 1
.venv-tune/bin/python -m selfheal.rollout --tasks out/splits.json --split pool --k 4 \
  --out out/r1/student.jsonl --tag r1-student --concurrency 48 --stop 2>&1 | grep -E "^\[rollout" | tail -1
echo "=== r1 student summary ==="
.venv-tune/bin/python -m selfheal.summary out/r1/student.jsonl --json out/r1/student.summary.json
echo "=== r1 buckets ==="
.venv-tune/bin/python -m selfheal.buckets out/r1/student.jsonl --out out/r1/buckets.json
echo "=== r1 rule labels ==="
.venv-tune/bin/python -m selfheal.failures out/r1/student.jsonl --out out/r1/failures_rules.json 2>&1 | grep -v Warning
