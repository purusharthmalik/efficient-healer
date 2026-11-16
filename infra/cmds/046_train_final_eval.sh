# Final LoRA (trained from the base model on the round 2 set), held-out evaluation in both
# harness modes, AuditKIT comparison, and the two gate decisions.
set -u
cd /work
export PYTHONPATH=/work/sync/code
PY=.venv-tune/bin/python
AK=.venv-audit/bin/python
[ -s out/r2/dataset/rows.jsonl ] || { echo "no round 2 rows"; exit 1; }
pkill -f "served-model-name student-v1 " 2>/dev/null; sleep 4
t0=$(date +%s)
$PY -m selfheal.train --rows out/r2/dataset/rows.jsonl --trajectories out/r2/dataset/trajectories.jsonl --out out/final/adapter \
  --loss agent_turns --epochs 3 --lr 2e-4 --lora-r 16 --lora-alpha 32 --batch-size 4 --grad-accum 4 2>&1 | grep -E "^\[train\]|'loss'|train_loss|Error|error|Traceback|CUDA|OOM" | cut -c1-330 | awk 'NR<=8 || NR%5==0'
echo "training took $(( $(date +%s) - t0 ))s"
[ -f out/final/adapter/adapter_model.safetensors ] || { echo "NO ADAPTER"; exit 1; }
$PY -m selfheal.merge --adapter out/final/adapter --out out/final/merged 2>&1 | grep -E "^\[merge\]|Error|Traceback"
bash /work/sync/code/cmds/serve_model.sh /work/out/final/merged student-final 8004 0.15 --dtype bfloat16 || exit 1
FIN="--model-path /work/out/final/merged --served-model student-final --base-url http://127.0.0.1:8004/v1"
for mode in stop nostop; do
  flag=""; [ "$mode" = stop ] && flag="--stop"
  $PY -m selfheal.rollout --tasks out/splits.json --split eval --k 3 --out out/eval/final_$mode.jsonl --tag final-$mode $FIN --concurrency 48 $flag 2>&1 | grep -E "^\[rollout" | tail -1
  echo "=== final_$mode ==="
  $PY -m selfheal.summary out/eval/final_$mode.jsonl --json out/eval/final_$mode.summary.json | grep -E '"success_rate|strict|tasks_|mean_tool_calls|mean_completion|mean_seconds"'
done

export XDG_CACHE_HOME=/work/.cache_audit
for pair in "base_stop v1_stop r1_gate" "v1_stop final_stop r2_gate" "base_stop final_stop overall_stop" "base_nostop final_nostop overall_nostop" "base_nostop v1_nostop r1_nostop"; do
  set -- $pair
  echo "=== AuditKIT: $1 vs $2 ==="
  $AK -m selfheal.evaluate --base out/eval/$1.jsonl --cand out/eval/$2.jsonl --out out/eval/cmp_$3 2>&1 | grep -v -E "Warning|warnings.warn" | tail -45
done
echo "=== gate, round 1: v1 vs base ==="
$PY -m selfheal.gate --report out/eval/cmp_r1_gate/report.json --base out/eval/base_stop.jsonl --cand out/eval/v1_stop.jsonl --label "round 1: v1 vs base" --out out/eval/gate_round1.json 2>&1 | grep -v -E "Warning|warnings.warn"
echo "=== gate, round 2: final vs v1 ==="
$PY -m selfheal.gate --report out/eval/cmp_r2_gate/report.json --base out/eval/v1_stop.jsonl --cand out/eval/final_stop.jsonl --label "round 2: final vs v1" --out out/eval/gate_round2.json 2>&1 | grep -v -E "Warning|warnings.warn"
