# Exploration bootstrap: environments, then the published run state back into /work/out.
set -u
cd /work
bash /work/sync/code/cmds/000_setup.sh 2>&1 | grep -E "rc=|tune:|vllm:|audit:|ok \("
export HF_XET_HIGH_PERFORMANCE=1
hf download Purusharth/efficient-healer-trajectories --repo-type dataset --local-dir dl --quiet >/dev/null 2>&1
hf download Purusharth/tiny-aya-efficient-healer-lora --local-dir adapters --quiet >/dev/null 2>&1
mkdir -p out/eval out/final out/r1
cp -r dl/round1/. out/r1/ ; cp -r dl/round2 out/r2 ; cp -r dl/eval/. out/eval/ ; cp dl/splits.json dl/splits_confirm.json out/
cp -r adapters/round2 out/final/adapter ; cp -r adapters/round1 out/r1/adapter
mkdir -p out/pilot && cp dl/pilot/teacher.jsonl out/pilot/ 2>/dev/null
ls out out/r1 out/r2 | head -40
du -sh out | cut -f1
export PYTHONPATH=/work/sync/code
.venv-tune/bin/python -m selfheal.merge --adapter out/final/adapter --out out/final/merged 2>&1 | grep -E "^\[merge\]|Error|Traceback"
