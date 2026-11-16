# Exploration: retention and safety check with AuditKIT (base Tiny Aya vs the deployed round 2 adapter).
set -u
cd /work
export PYTHONPATH=/work/sync/code
export XDG_CACHE_HOME=/work/.cache_audit
pkill -f "served-model-name expanse" 2>/dev/null; sleep 4      # frees the Expanse servers' memory
bash /work/sync/code/cmds/serve_student.sh 0.15 || exit 1
bash /work/sync/code/cmds/serve_model.sh /work/out/final/merged student-final 8004 0.15 --dtype bfloat16 || exit 1
t0=$(date +%s)
.venv-audit/bin/python -m selfheal.retention --base student http://127.0.0.1:8000/v1 --cand student-final http://127.0.0.1:8004/v1 \
  --n 100 --out out/eval/retention/report.json 2>&1 | grep -v -E "Warning|warnings.warn" | tail -70
echo "retention check took $(( $(date +%s) - t0 ))s"
