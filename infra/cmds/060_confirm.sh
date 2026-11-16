# One confirmatory evaluation on 400 fresh questions from the held-out companies (k=3, stop rule),
# for base, round 1 and final. The same gate rule is applied once; its answer stands.
set -u
cd /work
export PYTHONPATH=/work/sync/code
export XDG_CACHE_HOME=/work/.cache_audit
PY=.venv-tune/bin/python
AK=.venv-audit/bin/python
pkill -f "served-model-name teacher " 2>/dev/null; sleep 3            # the teacher is no longer needed
$PY -m selfheal.confirm_split --n 400 --out out/splits_confirm.json 2>&1 | grep -v -E "Warning|warnings.warn" || exit 1
( bash /work/sync/code/cmds/serve_model.sh /work/out/r1/merged student-v1 8003 0.15 --dtype bfloat16 > logs/serve_v1_again.out 2>&1 & )
bash /work/sync/code/cmds/serve_student.sh 0.15 || exit 1
bash /work/sync/code/cmds/serve_model.sh /work/out/final/merged student-final 8004 0.15 --dtype bfloat16 || exit 1
for i in $(seq 1 150); do curl -sf localhost:8003/health >/dev/null && break; sleep 2; done
curl -sf localhost:8003/health >/dev/null && echo "student-v1 server up" || { echo "V1 SERVER FAILED"; tail -5 logs/serve_v1_again.out; exit 1; }

run() { # tag, model path, served name, port
  $PY -m selfheal.rollout --tasks out/splits_confirm.json --split confirm --k 3 --stop --out out/eval/confirm_$1.jsonl --tag confirm-$1 \
    --model-path $2 --served-model $3 --base-url http://127.0.0.1:$4/v1 --concurrency 48 2>&1 | grep -E "^\[rollout" | tail -1
  echo "=== confirm_$1 ==="
  $PY -m selfheal.summary out/eval/confirm_$1.jsonl --json out/eval/confirm_$1.summary.json | grep -E '"n_tasks|"success_rate|strict|tasks_|mean_tool_calls|mean_completion'
}
run base CohereLabs/tiny-aya-global student 8000
run v1 /work/out/r1/merged student-v1 8003
run final /work/out/final/merged student-final 8004

for pair in "base v1 confirm_r1" "v1 final confirm_r2" "base final confirm_overall"; do
  set -- $pair
  $AK -m selfheal.evaluate --base out/eval/confirm_$1.jsonl --cand out/eval/confirm_$2.jsonl --out out/eval/cmp_$3 2>&1 | grep -E "Traceback|Error" | head -3
done
echo "=== confirmatory gate, round 1 ==="
$PY -m selfheal.gate --report out/eval/cmp_confirm_r1/report.json --base out/eval/confirm_base.jsonl --cand out/eval/confirm_v1.jsonl \
  --label "round 1, confirmatory set: v1 vs base" --out out/eval/gate_round1_confirm.json 2>&1 | grep -E '"deploy"|PASS|FAIL|approved' | head -8
echo "=== confirmatory gate, round 2 ==="
$PY -m selfheal.gate --report out/eval/cmp_confirm_r2/report.json --base out/eval/confirm_v1.jsonl --cand out/eval/confirm_final.jsonl \
  --label "round 2, confirmatory set: final vs v1" --out out/eval/gate_round2_confirm.json 2>&1 | grep -E '"deploy"|PASS|FAIL|approved' | head -8
echo "=== private upload with the confirmatory results ==="
$PY -m selfheal.publish --out out --model-repo Purusharth/tiny-aya-efficient-healer-lora \
  --dataset-repo Purusharth/efficient-healer-trajectories 2>&1 | grep -E "^\[publish\]|Error|Traceback" | cut -c1-240
