# Exploration A: round 3 on a pool of 1,400 tasks. Collect with the round 2 adapter, repair, build the set.
set -u
cd /work
export PYTHONPATH=/work/sync/code
export HF_XET_HIGH_PERFORMANCE=1
PY=.venv-tune/bin/python
Q() { grep -v -E "Warning|warnings.warn|^\s*$"; }
FIN="--model-path /work/out/final/merged --served-model student-final --base-url http://127.0.0.1:8004/v1"
bash /work/sync/code/cmds/serve_model.sh /work/out/final/merged student-final 8004 0.15 --dtype bfloat16 || exit 1
hf download nvidia/Qwen3.6-35B-A3B-NVFP4 --quiet >/dev/null 2>&1
bash /work/sync/code/cmds/serve_teacher.sh 0.45 || exit 1

echo "=== grow pool to 1,400 ==="
cp -n out/splits.json out/splits_480.json
$PY -m selfheal.tasks --n-pool 1400 --n-eval 100 --out out/splits.json 2>&1 | grep -E "pool_size|eval_size|shared_"
$PY - <<'PYEOF'
import json
old, new = json.load(open("out/splits_480.json")), json.load(open("out/splits.json"))
ids = lambda rows: [r["task_id"] for r in rows]
assert ids(old["eval"]) == ids(new["eval"]) and ids(old["pool"]) == ids(new["pool"])[:480]
print("split check ok: held-out set unchanged, the 480-task pool is a prefix of the new pool")
PYEOF
[ $? -eq 0 ] || exit 1

echo "=== a second fresh held-out set (B) for the round 3 gate, and the round 2 adapter on it ==="
$PY -m selfheal.confirm_split --n 400 --seed 2 --exclude out/splits_confirm.json --out out/splits_fresh2.json 2>&1 | Q
$PY -m selfheal.rollout --tasks out/splits_fresh2.json --split confirm --k 3 --stop --out out/eval/setB_final.jsonl --tag setB-final $FIN --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/eval/setB_final.jsonl --json out/eval/setB_final.summary.json | grep -E '"n_tasks|"success_rate|tasks_all'

echo "=== round 3 collection with the round 2 adapter (1,400 x 4) ==="
$PY -m selfheal.rollout --tasks out/splits.json --split pool --k 4 --stop --out out/r3/student.jsonl --tag r3-student $FIN --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/r3/student.jsonl --json out/r3/student.summary.json | grep -E '"n_tasks|"success_rate|strict|tasks_|mean_tool_calls'
$PY -m selfheal.buckets out/r3/student.jsonl --out out/r3/buckets.json | grep -E '"always|"sometimes|"never|"tasks|with_target|mean_calls'

echo "=== labels and hints ==="
$PY -m selfheal.failures out/r3/student.jsonl --out out/r3/failures.json --model openai/teacher --api-base http://127.0.0.1:8001/v1 --per-task 1 --never-only --no-thinking 2>&1 | Q | grep -E "llm_class|rule_vs|never_solved|Error"
$PY -m selfheal.repair --failures out/r3/failures.json --out out/r3/repair | Q | grep -E "never|hint" | head -3
$PY -m selfheal.pending --tasks out/splits.json --have out/r3/student.jsonl --out out/r3/repair/retry_ids.json
$PY -m selfheal.rollout --tasks out/splits.json --split pool --only out/r3/repair/retry_ids.json --hints out/r3/repair/hints.json --hint-label r3 \
  --k 4 --stop --out out/r3/hint_retry.jsonl --tag r3-hint $FIN --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/r3/hint_retry.jsonl --json out/r3/hint_retry.summary.json | grep -E '"n_tasks|"success_rate|tasks_'

echo "=== teacher runs for tasks with no verified run yet ==="
$PY -m selfheal.pending --tasks out/splits.json --have out/r3/student.jsonl out/r3/hint_retry.jsonl out/r1/teacher.jsonl out/r2/teacher.jsonl out/pilot/teacher.jsonl --out out/r3/repair/teacher_ids.json
$PY -m selfheal.rollout --tasks out/splits.json --split pool --only out/r3/repair/teacher_ids.json --k 3 \
  --model-path nvidia/Qwen3.6-35B-A3B-NVFP4 --served-model teacher --base-url http://127.0.0.1:8001/v1 --max-new-tokens 1024 \
  --out out/r3/teacher.jsonl --tag r3-teacher --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/r3/teacher.jsonl --json out/r3/teacher.summary.json | grep -E '"n_tasks|"success_rate|tasks_'

echo "=== round 3 training set ==="
$PY -m selfheal.build_dataset --student out/r3/student.jsonl --hint-retries out/r3/hint_retry.jsonl \
  --teacher out/r1/teacher.jsonl out/pilot/teacher.jsonl out/r2/teacher.jsonl out/r3/teacher.jsonl --round 3 --out out/r3/dataset 2>&1 | Q | grep -E "rows_|efficiency|reliability|repair_|tasks_without|dropped"
