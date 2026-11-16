# Exploration B3 (best effort): DPO on step-level preference pairs, on top of the round 2 adapter.
set -u
cd /work
export PYTHONPATH=/work/sync/code
export XDG_CACHE_HOME=/work/.cache_audit
PY=.venv-tune/bin/python
AK=.venv-audit/bin/python
$PY -m selfheal.dpo_pairs --rows out/r2/dataset/rows.jsonl --student out/r2/student.jsonl --out out/dpo/pairs.jsonl --max-pairs 480 2>&1 | grep -v -E "Warning|warnings.warn"
pkill -f "served-model-name" 2>/dev/null; sleep 5      # all servers down: DPO needs the memory and the compute
t0=$(date +%s)
$PY -m selfheal.train_dpo --pairs out/dpo/pairs.jsonl --model /work/out/final/merged --out out/dpo/adapter 2>&1 \
  | grep -E "^\[dpo\]|'loss'|train_loss|Error|error|Traceback|CUDA|OOM|TypeError|ValueError|KeyError" | cut -c1-360 | awk 'NR<=10 || NR%4==0'
echo "DPO training took $(( $(date +%s) - t0 ))s"
[ -f out/dpo/adapter/adapter_model.safetensors ] || { echo "NO DPO ADAPTER"; exit 1; }
hf upload Purusharth/selfheal-ops out/dpo/adapter adapters/dpo --repo-type dataset --quiet >/dev/null 2>&1 && echo "adapter saved to the private ops repo"
$PY -m selfheal.merge --base /work/out/final/merged --adapter out/dpo/adapter --out out/dpo/merged 2>&1 | grep -E "^\[merge\]|Error|Traceback"
bash /work/sync/code/cmds/serve_model.sh /work/out/dpo/merged student-dpo 8007 0.15 --dtype bfloat16 || exit 1
$PY -m selfheal.rollout --tasks out/splits_confirm.json --split confirm --k 3 --stop --out out/eval/confirm_dpo.jsonl --tag confirm-dpo \
  --model-path /work/out/dpo/merged --served-model student-dpo --base-url http://127.0.0.1:8007/v1 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
echo "=== confirm_dpo ==="
$PY -m selfheal.summary out/eval/confirm_dpo.jsonl --json out/eval/confirm_dpo.summary.json | grep -E '"n_tasks|"success_rate|strict|tasks_|mean_tool_calls|mean_completion'
$AK -m selfheal.evaluate --base out/eval/confirm_final.jsonl --cand out/eval/confirm_dpo.jsonl --out out/eval/cmp_dpo 2>&1 | grep -E "Traceback|Error" | head -3
$PY -m selfheal.gate --report out/eval/cmp_dpo/report.json --base out/eval/confirm_final.jsonl --cand out/eval/confirm_dpo.jsonl \
  --label "DPO on top of the round 2 adapter vs the round 2 adapter (confirmatory set)" --out out/eval/gate_dpo.json 2>&1 | grep -E '"deploy"|PASS|FAIL|approved' | head -8
