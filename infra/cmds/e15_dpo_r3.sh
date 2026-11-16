# Last step on job A (time-boxed): DPO on top of the round 3 adapter, gated on fresh set B.
set -u
cd /work
export PYTHONPATH=/work/sync/code
export XDG_CACHE_HOME=/work/.cache_audit
PY=.venv-tune/bin/python
AK=.venv-audit/bin/python
$PY -m selfheal.dpo_pairs --rows out/r3/dataset/rows.jsonl --student out/r3/student.jsonl --out out/dpo3/pairs.jsonl --max-pairs 320 2>&1 | grep -E "pairs_found|tasks_with|pairs_used"
pkill -f "served-model-name" 2>/dev/null; sleep 4
t0=$(date +%s)
$PY -m selfheal.train_dpo --pairs out/dpo3/pairs.jsonl --model /work/out/r3/merged --out out/dpo3/adapter 2>&1 \
  | grep -E "^\[dpo\]|Error|Traceback|CUDA|OOM|TypeError|ValueError" | cut -c1-300 | tail -6
echo "DPO training took $(( $(date +%s) - t0 ))s"
[ -f out/dpo3/adapter/adapter_model.safetensors ] || { echo "NO DPO ADAPTER"; exit 1; }
hf upload Purusharth/selfheal-ops out/dpo3/adapter adapters/round3_dpo --repo-type dataset --quiet >/dev/null 2>&1 && echo "adapter saved to the private ops repo"
$PY -m selfheal.merge --base /work/out/r3/merged --adapter out/dpo3/adapter --out out/dpo3/merged 2>&1 | grep -E "^\[merge\]|Error|Traceback"
bash /work/sync/code/cmds/serve_model.sh /work/out/dpo3/merged student-dpo3 8008 0.15 --dtype bfloat16 || exit 1
$PY -m selfheal.rollout --tasks out/splits_fresh2.json --split confirm --k 3 --stop --out out/eval/setB_r3dpo.jsonl --tag setB-r3dpo \
  --model-path /work/out/dpo3/merged --served-model student-dpo3 --base-url http://127.0.0.1:8008/v1 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
echo "=== setB_r3dpo ==="
$PY -m selfheal.summary out/eval/setB_r3dpo.jsonl --json out/eval/setB_r3dpo.summary.json | grep -E '"success_rate|strict|tasks_all|mean_tool_calls"'
$AK -m selfheal.evaluate --base out/eval/setB_r3.jsonl --cand out/eval/setB_r3dpo.jsonl --out out/eval/cmp_setB_r3dpo 2>&1 | grep -E "Traceback|Error" | head -3
$PY -m selfheal.gate --report out/eval/cmp_setB_r3dpo/report.json --base out/eval/setB_r3.jsonl --cand out/eval/setB_r3dpo.jsonl \
  --label "round 3 + DPO vs round 3 adapter, fresh held-out set B" --out out/eval/gate_r3dpo_setB.json 2>&1 | grep -E '"deploy"|PASS|FAIL' | head -5
