# The "before" numbers: base Tiny Aya on the 100 held-out tasks, k=3, with and without the stop rule.
set -u
cd /work
export PYTHONPATH=/work/sync/code
pkill -f "served-model-name expanse" 2>/dev/null; sleep 3   # frees 24 GB; Expanse is not used
bash /work/sync/code/cmds/serve_student.sh 0.15 || exit 1
for mode in stop nostop; do
  flag=""; [ "$mode" = stop ] && flag="--stop"
  .venv-tune/bin/python -m selfheal.rollout --tasks out/splits.json --split eval --k 3 \
    --out out/eval/base_$mode.jsonl --tag base-$mode --concurrency 48 $flag 2>&1 | grep -E "^\[rollout" | tail -1
  echo "=== base_$mode ==="
  .venv-tune/bin/python -m selfheal.summary out/eval/base_$mode.jsonl --json out/eval/base_$mode.summary.json
done
