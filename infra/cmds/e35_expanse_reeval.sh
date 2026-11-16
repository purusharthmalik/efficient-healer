# Redo the Aya Expanse evaluation: the first one ran without the FinQA database (every lookup
# returned "No tables found"), so its numbers are void.
set -u
cd /work
export PYTHONPATH=/work/sync/code
export XDG_CACHE_HOME=/work/.cache_audit
PY=.venv-tune/bin/python
AK=.venv-audit/bin/python
mkdir -p out/void && mv out/eval/confirm_expanse_* out/eval/cmp_expanse_vs_tiny out/void/ 2>/dev/null
rm -f finqa.db
pkill -f "served-model-name student" 2>/dev/null; sleep 4
bash /work/sync/code/cmds/serve_model.sh /work/out/expanse/merged expanse-ft 8006 0.30 --dtype bfloat16 || exit 1
$PY -m selfheal.rollout --tasks out/splits_confirm.json --split confirm --k 3 --stop --out out/eval/confirm_expanse_ft.jsonl \
  --tag confirm-expanse-ft --model-path /work/out/expanse/merged --served-model expanse-ft --base-url http://127.0.0.1:8006/v1 --concurrency 48 2>&1 | grep -E "^\[rollout|Assertion|Error" | tail -3
ls -la finqa.db | awk '{print "finqa.db bytes:", $5}'
echo "=== confirm_expanse_ft (valid) ==="
$PY -m selfheal.summary out/eval/confirm_expanse_ft.jsonl --json out/eval/confirm_expanse_ft.summary.json | grep -E '"n_tasks|"success_rate|strict|tasks_|mean_tool_calls|mean_completion|mean_seconds"'
echo "=== AuditKIT: Tiny Aya round 2 adapter vs fine-tuned Aya Expanse 8B, same 400 tasks ==="
$AK -m selfheal.evaluate --base out/eval/confirm_final.jsonl --cand out/eval/confirm_expanse_ft.jsonl --out out/eval/cmp_expanse_vs_tiny 2>&1 | grep -v -E "Warning|warnings.warn" | grep -A8 '"success_rate"' | head -10
pkill -f "served-model-name expanse-ft" 2>/dev/null; sleep 3
echo "=== untrained Aya Expanse 8B, first 100 confirmatory tasks (valid) ==="
bash /work/sync/code/cmds/serve_model.sh CohereLabs/aya-expanse-8b expanse 8002 0.25 --dtype bfloat16 || exit 1
$PY -m selfheal.rollout --tasks out/splits_confirm.json --split confirm --limit 100 --k 3 --stop --out out/eval/confirm_expanse_base.jsonl \
  --tag confirm-expanse-base --model-path CohereLabs/aya-expanse-8b --served-model expanse --base-url http://127.0.0.1:8002/v1 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/eval/confirm_expanse_base.jsonl --json out/eval/confirm_expanse_base.summary.json | grep -E '"n_tasks|"success_rate|tasks_'
