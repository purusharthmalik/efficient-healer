# Re-run the AgentTune detector and retraining trigger for round 1 (the first attempt hit a loader bug).
set -u
cd /work
export PYTHONPATH=/work/sync/code
.venv-tune/bin/python -m selfheal.agenttune_loop --student out/r1/student.jsonl --rows out/r1/dataset/rows.jsonl --failures out/r1/failures.json --out out/r1/dataset 2>&1 | grep -v -E "Warning|warnings.warn|^\s*$" | tail -30
