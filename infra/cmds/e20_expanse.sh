# Exploration B1: Aya Expanse 8B as the student, trained on the same final (round 2) training set.
set -u
cd /work
export PYTHONPATH=/work/sync/code
export HF_XET_HIGH_PERFORMANCE=1
export XDG_CACHE_HOME=/work/.cache_audit
PY=.venv-tune/bin/python
AK=.venv-audit/bin/python
hf download CohereLabs/aya-expanse-8b --quiet >/dev/null 2>&1

echo "=== untrained Aya Expanse 8B on the first 100 confirmatory tasks (k=3, stop rule) ==="
bash /work/sync/code/cmds/serve_model.sh CohereLabs/aya-expanse-8b expanse 8002 0.25 --dtype bfloat16 || exit 1
$PY -m selfheal.rollout --tasks out/splits_confirm.json --split confirm --limit 100 --k 3 --stop --out out/eval/confirm_expanse_base.jsonl \
  --tag confirm-expanse-base --model-path CohereLabs/aya-expanse-8b --served-model expanse --base-url http://127.0.0.1:8002/v1 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/eval/confirm_expanse_base.jsonl --json out/eval/confirm_expanse_base.summary.json | grep -E '"n_tasks|"success_rate|tasks_|mean_tool_calls|mean_completion'
pkill -f "served-model-name expanse " 2>/dev/null; sleep 5

echo "=== train a LoRA on Aya Expanse 8B with the round 2 training set ==="
t0=$(date +%s)
$PY -m selfheal.train --student CohereLabs/aya-expanse-8b --rows out/r2/dataset/rows.jsonl --trajectories out/r2/dataset/trajectories.jsonl \
  --out out/expanse/adapter --loss agent_turns --epochs 3 --lr 2e-4 --lora-r 16 --lora-alpha 32 --batch-size 4 --grad-accum 4 2>&1 \
  | grep -E "^\[train\]|'loss'|train_loss|Error|error|Traceback|CUDA|OOM" | cut -c1-330 | awk 'NR<=8 || NR%5==0'
echo "training took $(( $(date +%s) - t0 ))s"
[ -f out/expanse/adapter/adapter_model.safetensors ] || { echo "NO ADAPTER"; exit 1; }
hf upload Purusharth/selfheal-ops out/expanse/adapter adapters/expanse --repo-type dataset --quiet >/dev/null 2>&1 && echo "adapter saved to the private ops repo"
$PY -m selfheal.merge --base CohereLabs/aya-expanse-8b --adapter out/expanse/adapter --out out/expanse/merged 2>&1 | grep -E "^\[merge\]|Error|Traceback"
bash /work/sync/code/cmds/serve_model.sh /work/out/expanse/merged expanse-ft 8006 0.30 --dtype bfloat16 || exit 1

echo "=== fine-tuned Aya Expanse 8B on the 400-task confirmatory set (k=3, stop rule) ==="
$PY -m selfheal.rollout --tasks out/splits_confirm.json --split confirm --k 3 --stop --out out/eval/confirm_expanse_ft.jsonl \
  --tag confirm-expanse-ft --model-path /work/out/expanse/merged --served-model expanse-ft --base-url http://127.0.0.1:8006/v1 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/eval/confirm_expanse_ft.jsonl --json out/eval/confirm_expanse_ft.summary.json | grep -E '"n_tasks|"success_rate|strict|tasks_|mean_tool_calls|mean_completion|mean_seconds"'
echo "=== AuditKIT: Tiny Aya final adapter vs fine-tuned Aya Expanse 8B, same 400 tasks ==="
$AK -m selfheal.evaluate --base out/eval/confirm_final.jsonl --cand out/eval/confirm_expanse_ft.jsonl --out out/eval/cmp_expanse_vs_tiny 2>&1 | grep -v -E "Warning|warnings.warn" | grep -A8 '"success_rate"' | head -12
