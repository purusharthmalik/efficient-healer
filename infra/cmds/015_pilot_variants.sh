# Same 20 pilot tasks x 2, three variants, to decide the student and the harness stop rule.
set -u
cd /work
export PYTHONPATH=/work/sync/code
export HF_XET_HIGH_PERFORMANCE=1
run() { # tag, model path, served name, port, extra rollout args
  .venv-tune/bin/python -m selfheal.rollout --tasks out/splits.json --split pool --limit 20 --k 2 \
    --out out/pilot/$1.jsonl --tag $1 --model-path $2 --served-model $3 --base-url http://127.0.0.1:$4/v1 \
    --concurrency 32 ${5:-} 2>&1 | grep -E "^\[rollout" | tail -2
  echo "=== $1 ==="
  .venv-tune/bin/python -m selfheal.summary out/pilot/$1.jsonl --json out/pilot/$1.summary.json | grep -E 'success_rate|tasks_|mean_tool_calls|mean_completion|mean_seconds|mean_gen|no_answer|wrong_number|"match|pred_not|rollout_error'
}
bash /work/sync/code/cmds/serve_student.sh 0.15 || exit 1
run tiny_stop CohereLabs/tiny-aya-global student 8000 --stop

hf download CohereLabs/aya-expanse-8b --quiet >/dev/null 2>&1
bash /work/sync/code/cmds/serve_model.sh CohereLabs/aya-expanse-8b expanse 8002 0.25 --dtype bfloat16 || exit 1
run expanse_nostop CohereLabs/aya-expanse-8b expanse 8002
run expanse_stop CohereLabs/aya-expanse-8b expanse 8002 --stop
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
