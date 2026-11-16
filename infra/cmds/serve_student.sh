# Start the student vLLM server on port 8000 if it is not already up.
# Usage: bash serve_student.sh [gpu_fraction]
cd /work
FRACTION=${1:-0.15}
if ! curl -sf localhost:8000/health >/dev/null; then
  # The FlashInfer sampler needs a CUDA compiler toolchain that this image lacks.
  export VLLM_USE_FLASHINFER_SAMPLER=0
  setsid nohup .venv-vllm/bin/vllm serve CohereLabs/tiny-aya-global \
    --served-model-name student --port 8000 --max-model-len 8192 \
    --gpu-memory-utilization "$FRACTION" --dtype bfloat16 > logs/vllm_student.srv 2>&1 < /dev/null &
  PID=$!
  for i in $(seq 1 120); do
    curl -sf localhost:8000/health >/dev/null && break
    kill -0 $PID 2>/dev/null || break
    sleep 2
  done
fi
if curl -sf localhost:8000/health >/dev/null; then
  echo "student server up"
else
  echo "STUDENT SERVER FAILED"; grep -E "Error|error:|Exception" logs/vllm_student.srv | grep -v 'File "' | tail -8 | cut -c1-400; exit 1
fi
