# Retrain LoRA v1 with the loss on agent turns only. The first attempt (loss on all tokens,
# 2% success) is kept under out/neg_alltokens as a negative result.
set -u
cd /work
export PYTHONPATH=/work/sync/code
PY=.venv-tune/bin/python
Q() { grep -v -E "Warning|warnings.warn|^\s*$"; }

if [ ! -d out/neg_alltokens ]; then
  mkdir -p out/neg_alltokens
  mv out/eval/v1_stop.jsonl out/eval/v1_stop.summary.json out/neg_alltokens/ 2>/dev/null
  mv out/r2 out/neg_alltokens/r2 2>/dev/null
  mv out/r1/adapter out/neg_alltokens/adapter 2>/dev/null
fi
pkill -f "served-model-name student-v1 " 2>/dev/null; sleep 4
rm -rf out/r1/merged

echo "=== round 1 training set, exact repeats removed ==="
$PY -m selfheal.build_dataset --student out/r1/student.jsonl --hint-retries out/r1/hint_retry.jsonl --teacher out/r1/teacher.jsonl out/pilot/teacher.jsonl --round 1 --out out/r1/dataset 2>&1 | Q | grep -E "rows_|repair_|reliability|tasks_without|dropped"

t0=$(date +%s)
$PY -m selfheal.train --rows out/r1/dataset/rows.jsonl --trajectories out/r1/dataset/trajectories.jsonl --out out/r1/adapter \
  --loss agent_turns --epochs 3 --lr 2e-4 --lora-r 16 --lora-alpha 32 --batch-size 4 --grad-accum 4 2>&1 | grep -E "^\[train\]|'loss'|train_loss|Error|error|Traceback|CUDA|OOM" | cut -c1-520 | awk 'NR<=9 || NR%5==0'
echo "training took $(( $(date +%s) - t0 ))s"
[ -f out/r1/adapter/adapter_model.safetensors ] || { echo "NO ADAPTER"; exit 1; }
$PY -m selfheal.merge --adapter out/r1/adapter --out out/r1/merged 2>&1 | grep -E "^\[merge\]|Error|Traceback"
bash /work/sync/code/cmds/serve_model.sh /work/out/r1/merged student-v1 8003 0.15 --dtype bfloat16 || exit 1
V1="--model-path /work/out/r1/merged --served-model student-v1 --base-url http://127.0.0.1:8003/v1"

echo "=== v1 on the held-out set (k=3, stop rule) ==="
$PY -m selfheal.rollout --tasks out/splits.json --split eval --k 3 --stop --out out/eval/v1_stop.jsonl --tag v1-stop $V1 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/eval/v1_stop.jsonl --json out/eval/v1_stop.summary.json | grep -E '"success_rate|strict|tasks_|mean_tool_calls|mean_completion|no_answer_tag|wrong_number|"match|pred_not'

echo "=== round 2 collection with v1 on the pool (k=4, stop rule) ==="
$PY -m selfheal.rollout --tasks out/splits.json --split pool --k 4 --stop --out out/r2/student.jsonl --tag r2-student $V1 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/r2/student.jsonl --json out/r2/student.summary.json | grep -E '"n_tasks|"success_rate|strict|tasks_|mean_tool_calls|mean_completion'
$PY -m selfheal.buckets out/r2/student.jsonl --out out/r2/buckets.json | grep -E '"always|"sometimes|"never|"tasks|with_target|mean_calls'
