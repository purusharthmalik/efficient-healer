cd /work
echo "--- first error lines ---"
grep -n -i -E "error|exception|unsupported|not supported|cohere|No module|cannot|failed" logs/vllm_student.srv | grep -v "^.*File \"" | head -40
echo "--- context around first Traceback ---"
L=$(grep -n "Traceback" logs/vllm_student.srv | head -1 | cut -d: -f1)
[ -n "$L" ] && sed -n "$((L-12)),$((L+45))p" logs/vllm_student.srv | cut -c1-400
echo "--- gpu ---"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
