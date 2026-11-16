# Exploration A: train the round 3 LoRA on the bigger set, evaluate it on a fresh held-out set (B)
# and on the confirmatory set (A), and apply the same gate against the round 2 adapter.
set -u
cd /work
export PYTHONPATH=/work/sync/code
export XDG_CACHE_HOME=/work/.cache_audit
PY=.venv-tune/bin/python
AK=.venv-audit/bin/python
[ -s out/r3/dataset/rows.jsonl ] || { echo "no round 3 rows"; exit 1; }
echo "=== AgentTune detector and retraining trigger, round 3 ==="
$PY -m selfheal.agenttune_loop --student out/r3/student.jsonl --rows out/r3/dataset/rows.jsonl --failures out/r3/failures.json --out out/r3/dataset 2>&1 | grep -E "failures_detected|examples_buffered|trigger_fired|trigger_reason|dominant|Error"
pkill -f "served-model-name teacher " 2>/dev/null; pkill -f "served-model-name student-final " 2>/dev/null; sleep 5
t0=$(date +%s)
$PY -m selfheal.train --rows out/r3/dataset/rows.jsonl --trajectories out/r3/dataset/trajectories.jsonl --out out/r3/adapter \
  --loss agent_turns --epochs 3 --lr 2e-4 --lora-r 16 --lora-alpha 32 --batch-size 4 --grad-accum 4 2>&1 \
  | grep -E "^\[train\]|'loss'|train_loss|Error|error|Traceback|CUDA|OOM" | cut -c1-330 | awk 'NR<=8 || NR%8==0'
echo "training took $(( $(date +%s) - t0 ))s"
[ -f out/r3/adapter/adapter_model.safetensors ] || { echo "NO ADAPTER"; exit 1; }
hf upload Purusharth/selfheal-ops out/r3/adapter adapters/round3 --repo-type dataset --quiet >/dev/null 2>&1 && echo "adapter saved to the private ops repo"
$PY -m selfheal.merge --adapter out/r3/adapter --out out/r3/merged 2>&1 | grep -E "^\[merge\]|Error|Traceback"
bash /work/sync/code/cmds/serve_model.sh /work/out/r3/merged student-r3 8005 0.15 --dtype bfloat16 || exit 1
R3="--model-path /work/out/r3/merged --served-model student-r3 --base-url http://127.0.0.1:8005/v1"
for set in "setB out/splits_fresh2.json"; do
  set -- $set
  $PY -m selfheal.rollout --tasks $2 --split confirm --k 3 --stop --out out/eval/$1_r3.jsonl --tag $1-r3 $R3 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
  echo "=== $1_r3 ==="
  $PY -m selfheal.summary out/eval/$1_r3.jsonl --json out/eval/$1_r3.summary.json | grep -E '"n_tasks|"success_rate|strict|tasks_|mean_tool_calls|mean_completion'
done
$AK -m selfheal.evaluate --base out/eval/setB_final.jsonl --cand out/eval/setB_r3.jsonl --out out/eval/cmp_r3_gate 2>&1 | grep -E "Traceback|Error" | head -3
echo "=== gate, round 3: round 3 adapter vs round 2 adapter, fresh set B ==="
$PY -m selfheal.gate --report out/eval/cmp_r3_gate/report.json --base out/eval/setB_final.jsonl --cand out/eval/setB_r3.jsonl \
  --label "round 3: round 3 adapter vs round 2 adapter (fresh held-out set B, 400 tasks)" --out out/eval/gate_round3.json 2>&1 | grep -E '"deploy"|PASS|FAIL|approved' | head -8
