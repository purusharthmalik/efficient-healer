# Round 1: re-score with the corrected verifier, grow the pool to 480, label failures,
# repair (hinted student retries, then teacher), build the round 1 training set.
set -u
cd /work
export PYTHONPATH=/work/sync/code
PY=.venv-tune/bin/python
Q() { grep -v -E "Warning|warnings.warn|^\s*$"; }

echo "=== re-score with corrected verifier ==="
$PY -m selfheal.rescore out/eval/base_stop.jsonl out/eval/base_nostop.jsonl out/r1/student.jsonl out/pilot/*.jsonl out/pool/rollouts.jsonl 2>&1 | Q
for m in stop nostop; do $PY -m selfheal.summary out/eval/base_$m.jsonl --json out/eval/base_$m.summary.json | grep -E '"success_rate|tasks_all_k|tasks_some'; done

echo "=== grow pool to 480 ==="
cp -n out/splits.json out/splits_240.json
$PY -m selfheal.tasks --n-pool 480 --n-eval 100 --out out/splits.json 2>&1 | grep -E "pool_size|eval_size|shared_"
$PY - <<'PYEOF'
import json
old, new = json.load(open("out/splits_240.json")), json.load(open("out/splits.json"))
ids = lambda rows: [r["task_id"] for r in rows]
assert ids(old["eval"]) == ids(new["eval"]), "held-out set changed"
assert ids(old["pool"]) == ids(new["pool"])[:240], "old pool is not a prefix of the new pool"
print("split check ok: held-out set unchanged, old pool is a prefix of the new pool")
PYEOF
[ $? -eq 0 ] || exit 1

bash /work/sync/code/cmds/serve_student.sh 0.15 || exit 1
bash /work/sync/code/cmds/serve_teacher.sh 0.45 || exit 1

echo "=== student rollouts on the new tasks ==="
$PY -m selfheal.rollout --tasks out/splits.json --split pool --k 4 --stop --out out/r1/student.jsonl --tag r1-student --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/r1/student.jsonl --json out/r1/student.summary.json | grep -E '"n_tasks|"success_rate|tasks_|mean_tool_calls"'
$PY -m selfheal.buckets out/r1/student.jsonl --out out/r1/buckets.json | Q

echo "=== failure labels: rules + AgentTune FailureClassifier ==="
$PY -m selfheal.failures out/r1/student.jsonl --out out/r1/failures.json --model openai/teacher --api-base http://127.0.0.1:8001/v1 --per-task 1 2>&1 | Q | grep -v "^\[classify\] [0-9]*[1-9]/" 
$PY -m selfheal.repair --failures out/r1/failures.json --out out/r1/repair | Q

echo "=== repair step 1: hinted student retries ==="
$PY -m selfheal.pending --tasks out/splits.json --have out/r1/student.jsonl --out out/r1/repair/retry_ids.json
$PY -m selfheal.rollout --tasks out/splits.json --split pool --only out/r1/repair/retry_ids.json --hints out/r1/repair/hints.json --hint-label r1 \
  --k 4 --stop --out out/r1/hint_retry.jsonl --tag r1-hint --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/r1/hint_retry.jsonl --json out/r1/hint_retry.summary.json | grep -E '"n_tasks|"success_rate|tasks_'

echo "=== repair step 2: teacher rollouts ==="
$PY -m selfheal.pending --tasks out/splits.json --have out/r1/student.jsonl out/r1/hint_retry.jsonl --out out/r1/repair/teacher_ids.json
$PY -m selfheal.rollout --tasks out/splits.json --split pool --only out/r1/repair/teacher_ids.json --k 3 \
  --model-path nvidia/Qwen3.6-35B-A3B-NVFP4 --served-model teacher --base-url http://127.0.0.1:8001/v1 --max-new-tokens 1024 \
  --out out/r1/teacher.jsonl --tag r1-teacher --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/r1/teacher.jsonl --json out/r1/teacher.summary.json | grep -E '"n_tasks|"success_rate|strict|tasks_|mean_tool_calls|mean_seconds"'

echo "=== round 1 training set ==="
$PY -m selfheal.build_dataset --student out/r1/student.jsonl --hint-retries out/r1/hint_retry.jsonl --teacher out/r1/teacher.jsonl out/pilot/teacher.jsonl --round 1 --out out/r1/dataset 2>&1 | Q
echo "=== AgentTune detector and retraining trigger ==="
$PY -m selfheal.agenttune_loop --student out/r1/student.jsonl --rows out/r1/dataset/rows.jsonl --failures out/r1/failures.json --out out/r1/dataset 2>&1 | Q
