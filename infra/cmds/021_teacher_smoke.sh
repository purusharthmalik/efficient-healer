# Can the chosen teacher (NVFP4 Qwen3.6 35B) be served on this GPU next to the student?
set -u
cd /work
export HF_XET_HIGH_PERFORMANCE=1
t0=$(date +%s)
hf download nvidia/Qwen3.6-35B-A3B-NVFP4 --quiet >/dev/null 2>&1
echo "teacher downloaded ($(( $(date +%s) - t0 ))s)"
t0=$(date +%s)
bash /work/sync/code/cmds/serve_teacher.sh 0.40 || { echo "--- last server lines ---"; tail -25 logs/vllm_teacher.srv | cut -c1-300; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; exit 1; }
echo "teacher load time $(( $(date +%s) - t0 ))s"
t0=$(date +%s%N)
curl -s localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"teacher","messages":[{"role":"user","content":"A company had revenue of 1250 in 2019 and 1400 in 2020. What was the percentage growth? Explain in three sentences, then give the number."}],"max_tokens":200,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
 | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'choices' not in d: print('API ERROR', str(d)[:600]); sys.exit(1)
m = d['choices'][0]['message']; print('content:', (m.get('content') or '')[:500]); print('usage:', d.get('usage'))"
echo "one request took $(( ( $(date +%s%N) - t0 ) / 1000000 )) ms"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
