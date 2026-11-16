# Teacher on the same 20 pilot tasks x 2: how many can it solve, and how fast?
set -u
cd /work
export PYTHONPATH=/work/sync/code
bash /work/sync/code/cmds/serve_teacher.sh 0.45 || exit 1
.venv-tune/bin/python -m selfheal.rollout --tasks out/splits.json --split pool --limit 20 --k 2 \
  --out out/pilot/teacher.jsonl --tag teacher --model-path nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --served-model teacher --base-url http://127.0.0.1:8001/v1 --concurrency 32 --max-new-tokens 1024 2>&1 \
  | grep -E "^\[rollout|Error|error" | tail -4
echo "=== teacher ==="
.venv-tune/bin/python -m selfheal.summary out/pilot/teacher.jsonl --json out/pilot/teacher.summary.json --show 1 | cut -c1-900
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
