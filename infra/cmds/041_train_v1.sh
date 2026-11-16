# Round 1 training: LoRA v1 on the round 1 set, then merge it and serve it for round 2.
set -u
cd /work
export PYTHONPATH=/work/sync/code
PY=.venv-tune/bin/python
[ -s out/r1/dataset/rows.jsonl ] || { echo "no round 1 rows"; exit 1; }
pkill -f "served-model-name student " 2>/dev/null; sleep 4      # base student server no longer needed; frees GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
t0=$(date +%s)
$PY -m selfheal.train --rows out/r1/dataset/rows.jsonl --trajectories out/r1/dataset/trajectories.jsonl --out out/r1/adapter \
  --epochs 3 --lr 2e-4 --lora-r 16 --lora-alpha 32 --batch-size 4 --grad-accum 4 2>&1 | grep -E "^\[train\]|'loss'|train_loss|Error|error|Traceback|CUDA|OOM" | cut -c1-260 | awk 'NR<=6 || NR%6==0'
echo "training took $(( $(date +%s) - t0 ))s"
ls -la out/r1/adapter | head -12
[ -f out/r1/adapter/adapter_model.safetensors ] || { echo "NO ADAPTER"; exit 1; }
$PY -m selfheal.merge --adapter out/r1/adapter --out out/r1/merged 2>&1 | grep -E "^\[merge\]|Error|Traceback"
bash /work/sync/code/cmds/serve_model.sh /work/out/r1/merged student-v1 8003 0.15 --dtype bfloat16 || exit 1

echo "=== v1 on the held-out set (k=3, stop rule) ==="
$PY -m selfheal.rollout --tasks out/splits.json --split eval --k 3 --stop --out out/eval/v1_stop.jsonl --tag v1-stop \
  --model-path /work/out/r1/merged --served-model student-v1 --base-url http://127.0.0.1:8003/v1 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/eval/v1_stop.jsonl --json out/eval/v1_stop.summary.json

echo "=== round 2 collection with v1 on the pool (k=4, stop rule) ==="
$PY -m selfheal.rollout --tasks out/splits.json --split pool --k 4 --stop --out out/r2/student.jsonl --tag r2-student \
  --model-path /work/out/r1/merged --served-model student-v1 --base-url http://127.0.0.1:8003/v1 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
$PY -m selfheal.summary out/r2/student.jsonl --json out/r2/student.summary.json | grep -E '"n_tasks|"success_rate|strict|tasks_|mean_tool_calls|mean_completion'
$PY -m selfheal.buckets out/r2/student.jsonl --out out/r2/buckets.json
