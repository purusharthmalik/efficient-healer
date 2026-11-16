# Three isolated environments. They only exchange files, so their torch and
# transformers pins never have to agree.
#   .venv-tune   AgentTune (rollout loop, tools, rewards, classifier, trainer, gate)
#   .venv-vllm   vLLM server (fast generation; >=0.30 is the first with Cohere support)
#   .venv-audit  AuditKIT (metrics, comparison)
set -u
cd /work
AGENTTUNE_SHA=8918cc359d51aad35bb05f457d11cc4e5766d7eb

t0=$(date +%s)
uv venv -q .venv-tune --python 3.12
uv pip install -q --python .venv-tune/bin/python \
  "agenttune @ git+https://github.com/Lexsi-Labs/AgentTune@${AGENTTUNE_SHA}" hf_transfer openai
echo "tune rc=$? ($(( $(date +%s) - t0 ))s)"

t0=$(date +%s)
uv venv -q .venv-vllm --python 3.12
uv pip install -q --python .venv-vllm/bin/python "vllm>=0.30" hf_transfer
echo "vllm rc=$? ($(( $(date +%s) - t0 ))s)"

t0=$(date +%s)
uv venv -q .venv-audit --python 3.12
uv pip install -q --python .venv-audit/bin/python "auditkit[transformers,interop,jsonschema]==1.1.12"
echo "audit rc=$? ($(( $(date +%s) - t0 ))s)"

echo "--- versions ---"
.venv-tune/bin/python -c "import agenttune, torch, transformers, trl, peft; print('tune: agenttune', getattr(agenttune,'__version__','?'), '| torch', torch.__version__, '| transformers', transformers.__version__, '| trl', trl.__version__, '| peft', peft.__version__, '| cuda', torch.cuda.is_available())"
.venv-vllm/bin/python -c "import vllm, torch, transformers; print('vllm:', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
.venv-audit/bin/python -c "import auditkit, transformers; print('audit: auditkit', auditkit.__version__, '| transformers', transformers.__version__)"

echo "--- model download ---"
export HF_HUB_ENABLE_HF_TRANSFER=1
t0=$(date +%s)
hf download CohereLabs/tiny-aya-global --quiet >/dev/null && echo "tiny-aya-global ok ($(( $(date +%s) - t0 ))s)"
df -h /work | tail -1
