# Upload adapters and all run data to two PRIVATE Hugging Face repos.
set -u
cd /work
export PYTHONPATH=/work/sync/code
.venv-tune/bin/python -m selfheal.publish --out out --model-repo Purusharth/tiny-aya-efficient-healer-lora \
  --dataset-repo Purusharth/efficient-healer-trajectories 2>&1 | grep -E "^\[publish\]|Error|Traceback" | cut -c1-240
