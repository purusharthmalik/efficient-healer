# Start a vLLM server for any model if its port is not already serving.
# Usage: bash serve_model.sh MODEL SERVED_NAME PORT GPU_FRACTION [extra vllm args...]
cd /work
MODEL=$1; NAME=$2; PORT=$3; FRACTION=$4; shift 4
if ! curl -sf localhost:$PORT/health >/dev/null; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  setsid nohup .venv-vllm/bin/vllm serve "$MODEL" --served-model-name "$NAME" --port "$PORT" \
    --max-model-len 8192 --gpu-memory-utilization "$FRACTION" "$@" > "logs/vllm_$NAME.srv" 2>&1 < /dev/null &
  PID=$!
  for i in $(seq 1 240); do
    curl -sf localhost:$PORT/health >/dev/null && break
    kill -0 $PID 2>/dev/null || break
    sleep 2
  done
fi
if curl -sf localhost:$PORT/health >/dev/null; then echo "$NAME server up"; else
  echo "$NAME SERVER FAILED"; grep -E "Error|error:|Exception" "logs/vllm_$NAME.srv" | grep -v 'File "' | tail -8 | cut -c1-400; exit 1; fi
