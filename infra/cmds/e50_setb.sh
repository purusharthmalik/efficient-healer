# Clean re-test on the second fresh held-out set (B, 400 tasks never used for any decision on this job):
# DPO adapter and fine-tuned Aya Expanse 8B, each against the round 2 adapter's runs on the same set.
set -u
cd /work
export PYTHONPATH=/work/sync/code
export XDG_CACHE_HOME=/work/.cache_audit
PY=.venv-tune/bin/python
AK=.venv-audit/bin/python
$PY -m selfheal.confirm_split --n 400 --seed 2 --exclude out/splits_confirm.json --out out/splits_fresh2.json 2>&1 | grep -E "confirm_size|shared_"
hf download Purusharth/selfheal-ops --repo-type dataset --include "outa/eval/setB_final.jsonl" --include "outa/splits_fresh2.json" --local-dir fromA --quiet >/dev/null 2>&1
cp fromA/outa/eval/setB_final.jsonl out/eval/setB_final.jsonl
$PY - <<'PYEOF'
import json
a = [r["task_id"] for r in json.load(open("fromA/outa/splits_fresh2.json"))["confirm"]]
b = [r["task_id"] for r in json.load(open("out/splits_fresh2.json"))["confirm"]]
assert a == b, "set B differs between the two jobs"
print("set B is identical on both jobs:", len(b), "tasks")
PYEOF
[ $? -eq 0 ] || exit 1
$PY -m selfheal.rollout --tasks out/splits_fresh2.json --split confirm --k 3 --stop --out out/eval/setB_dpo.jsonl --tag setB-dpo \
  --model-path /work/out/dpo/merged --served-model student-dpo --base-url http://127.0.0.1:8007/v1 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
echo "=== setB_dpo ==="
$PY -m selfheal.summary out/eval/setB_dpo.jsonl --json out/eval/setB_dpo.summary.json | grep -E '"success_rate|strict|tasks_all|mean_tool_calls"'
$AK -m selfheal.evaluate --base out/eval/setB_final.jsonl --cand out/eval/setB_dpo.jsonl --out out/eval/cmp_setB_dpo 2>&1 | grep -E "Traceback|Error" | head -3
$PY -m selfheal.gate --report out/eval/cmp_setB_dpo/report.json --base out/eval/setB_final.jsonl --cand out/eval/setB_dpo.jsonl \
  --label "DPO vs round 2 adapter, fresh held-out set B" --out out/eval/gate_dpo_setB.json 2>&1 | grep -E '"deploy"|PASS|FAIL' | head -5
pkill -f "served-model-name student-dpo" 2>/dev/null; sleep 3
bash /work/sync/code/cmds/serve_model.sh /work/out/expanse/merged expanse-ft 8006 0.30 --dtype bfloat16 || exit 1
$PY -m selfheal.rollout --tasks out/splits_fresh2.json --split confirm --k 3 --stop --out out/eval/setB_expanse_ft.jsonl --tag setB-expanse-ft \
  --model-path /work/out/expanse/merged --served-model expanse-ft --base-url http://127.0.0.1:8006/v1 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
echo "=== setB_expanse_ft ==="
$PY -m selfheal.summary out/eval/setB_expanse_ft.jsonl --json out/eval/setB_expanse_ft.summary.json | grep -E '"success_rate|strict|tasks_all|mean_tool_calls"'
$AK -m selfheal.evaluate --base out/eval/setB_final.jsonl --cand out/eval/setB_expanse_ft.jsonl --out out/eval/cmp_setB_expanse 2>&1 | grep -v -E "Warning|warnings.warn" | grep -A8 '"success_rate"' | head -10
