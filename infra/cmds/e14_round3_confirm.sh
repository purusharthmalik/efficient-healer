# Round 3 adapter on the confirmatory set (A), so its number sits beside the published round 2 result.
set -u
cd /work
export PYTHONPATH=/work/sync/code
export XDG_CACHE_HOME=/work/.cache_audit
PY=.venv-tune/bin/python
AK=.venv-audit/bin/python
bash /work/sync/code/cmds/serve_model.sh /work/out/r3/merged student-r3 8005 0.15 --dtype bfloat16 || exit 1
$PY -m selfheal.rollout --tasks out/splits_confirm.json --split confirm --k 3 --stop --out out/eval/confirm_r3.jsonl --tag confirm-r3 \
  --model-path /work/out/r3/merged --served-model student-r3 --base-url http://127.0.0.1:8005/v1 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
echo "=== confirm_r3 ==="
$PY -m selfheal.summary out/eval/confirm_r3.jsonl --json out/eval/confirm_r3.summary.json | grep -E '"n_tasks|"success_rate|strict|tasks_|mean_tool_calls|mean_completion'
$AK -m selfheal.evaluate --base out/eval/confirm_final.jsonl --cand out/eval/confirm_r3.jsonl --out out/eval/cmp_confirm_r3 2>&1 | grep -E "Traceback|Error" | head -3
