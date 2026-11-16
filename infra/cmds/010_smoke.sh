# Smoke test: vLLM server for the student, FinQA data, 6 tasks x 2 rollouts,
# then check that AuditKIT can read the trajectories.
set -u
cd /work
export PYTHONPATH=/work/sync/code
export HF_HUB_ENABLE_HF_TRANSFER=1

if ! curl -sf localhost:8000/health >/dev/null; then
  setsid nohup .venv-vllm/bin/vllm serve CohereLabs/tiny-aya-global \
    --served-model-name student --port 8000 --max-model-len 8192 \
    --gpu-memory-utilization 0.40 --dtype bfloat16 > logs/vllm_student.srv 2>&1 < /dev/null &
  for i in $(seq 1 100); do curl -sf localhost:8000/health >/dev/null && break; sleep 3; done
fi
if curl -sf localhost:8000/health >/dev/null; then echo "student server up"; else echo "SERVER FAILED"; tail -40 logs/vllm_student.srv; exit 1; fi

.venv-tune/bin/python -m selfheal.tasks --n-pool 240 --n-eval 100 --out out/splits.json 2>&1 | grep -v Warning || exit 1

rm -f out/smoke/rollouts.jsonl
.venv-tune/bin/python -m selfheal.rollout --tasks out/splits.json --split pool --limit 6 --k 2 \
  --out out/smoke/rollouts.jsonl --tag smoke --concurrency 12 2>&1 | grep -v -E "Warning|warnings.warn" | tail -40

.venv-tune/bin/python -m selfheal.summary out/smoke/rollouts.jsonl --show 2

echo "--- auditkit inspect ---"
.venv-audit/bin/auditkit agent import-agenttune out/smoke/rollouts.jsonl --inspect 2>&1 | tail -30
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
