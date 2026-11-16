# Round 2: v1's own rollouts are bucketed (always / sometimes / never); tasks without a grounded
# success get labels, hinted v1 retries, then teacher runs. Builds the final training set.
set -u
cd /work
export PYTHONPATH=/work/sync/code
PY=.venv-tune/bin/python
Q() { grep -v -E "Warning|warnings.warn|^\s*$"; }
V1="--model-path /work/out/r1/merged --served-model student-v1 --base-url http://127.0.0.1:8003/v1"
bash /work/sync/code/cmds/serve_model.sh /work/out/r1/merged student-v1 8003 0.15 --dtype bfloat16 || exit 1
bash /work/sync/code/cmds/serve_teacher.sh 0.45 || exit 1

echo "=== v1 on the held-out set without the stop rule (k=3) ==="
$PY -m selfheal.rollout --tasks out/splits.json --split eval --k 3 --out out/eval/v1_nostop.jsonl --tag v1-nostop $V1 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/eval/v1_nostop.jsonl --json out/eval/v1_nostop.summary.json | grep -E '"success_rate|strict|tasks_|mean_tool_calls|mean_completion'

echo "=== round 2 failure labels ==="
$PY -m selfheal.failures out/r2/student.jsonl --out out/r2/failures.json --model openai/teacher --api-base http://127.0.0.1:8001/v1 --per-task 1 --never-only 2>&1 | Q | grep -v "^\[classify\]"
$PY -m selfheal.repair --failures out/r2/failures.json --out out/r2/repair | Q

echo "=== repair step 1: hinted v1 retries ==="
$PY -m selfheal.pending --tasks out/splits.json --have out/r2/student.jsonl --out out/r2/repair/retry_ids.json
$PY -m selfheal.rollout --tasks out/splits.json --split pool --only out/r2/repair/retry_ids.json --hints out/r2/repair/hints.json --hint-label r2 \
  --k 4 --stop --out out/r2/hint_retry.jsonl --tag r2-hint $V1 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/r2/hint_retry.jsonl --json out/r2/hint_retry.summary.json | grep -E '"n_tasks|"success_rate|tasks_'

echo "=== repair step 2: teacher runs (round 1 teacher runs are reused; new runs only for tasks the teacher has not tried) ==="
$PY -m selfheal.pending --tasks out/splits.json --have out/r2/student.jsonl out/r2/hint_retry.jsonl out/r1/teacher.jsonl out/pilot/teacher.jsonl --out out/r2/repair/teacher_ids.json
$PY -m selfheal.rollout --tasks out/splits.json --split pool --only out/r2/repair/teacher_ids.json --k 3 \
  --model-path nvidia/Qwen3.6-35B-A3B-NVFP4 --served-model teacher --base-url http://127.0.0.1:8001/v1 --max-new-tokens 1024 \
  --out out/r2/teacher.jsonl --tag r2-teacher --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1

echo "=== final training set ==="
$PY -m selfheal.build_dataset --student out/r2/student.jsonl --hint-retries out/r2/hint_retry.jsonl \
  --teacher out/r1/teacher.jsonl out/pilot/teacher.jsonl out/r2/teacher.jsonl --round 2 --out out/r2/dataset 2>&1 | Q
echo "=== AgentTune detector and retraining trigger ==="
$PY -m selfheal.agenttune_loop --student out/r2/student.jsonl --rows out/r2/dataset/rows.jsonl --failures out/r2/failures.json --out out/r2/dataset 2>&1 | Q
