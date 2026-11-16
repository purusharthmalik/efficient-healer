# De-risk the evaluation path before it matters: AuditKIT import + metrics + compare + gate,
# on files that already exist (base without stop rule as "base", base with stop rule as "candidate").
set -u
cd /work
export PYTHONPATH=/work/sync/code
export XDG_CACHE_HOME=/work/.cache_audit
.venv-audit/bin/python -m selfheal.evaluate --base out/eval/base_nostop.jsonl --cand out/eval/base_stop.jsonl --out out/eval/selftest 2>&1 | grep -v -E "Warning|warnings.warn" | tail -40
.venv-tune/bin/python -m selfheal.gate --report out/eval/selftest/report.json --base out/eval/base_nostop.jsonl --cand out/eval/base_stop.jsonl --label "self-test" --out out/eval/selftest/gate.json 2>&1 | grep -v -E "Warning|warnings.warn" | tail -30
