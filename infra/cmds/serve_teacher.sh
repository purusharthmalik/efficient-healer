# Start the teacher vLLM server on port 8001 if it is not already up.
# Usage: bash serve_teacher.sh [gpu_fraction]
cd /work
FRACTION=${1:-0.45}
MODEL=nvidia/Qwen3.6-35B-A3B-NVFP4
if ! curl -sf localhost:8001/health >/dev/null; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  setsid nohup .venv-vllm/bin/vllm serve "$MODEL" \
    --served-model-name teacher --port 8001 --quantization modelopt \
    --max-model-len 16384 --gpu-memory-utilization "$FRACTION" --max-num-seqs 64 \
    --reasoning-parser qwen3 > logs/vllm_teacher.srv 2>&1 < /dev/null &
  PID=$!
  for i in $(seq 1 300); do
    curl -sf localhost:8001/health >/dev/null && break
    kill -0 $PID 2>/dev/null || break
    sleep 2
  done
fi
if curl -sf localhost:8001/health >/dev/null; then
  echo "teacher server up"
else
  echo "TEACHER SERVER FAILED"
  grep -E "Error|error:|Exception|not supported|unsupported" logs/vllm_teacher.srv | grep -v 'File "' | tail -12 | cut -c1-500
  exit 1
fi
