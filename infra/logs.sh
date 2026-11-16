#!/usr/bin/env bash
# Print the last N lines of the GPU worker's log (the CLI's `hf jobs logs` hangs in this version).
#   infra/logs.sh [N] [seconds-to-listen]
JOB=${JOB:-$(grep -o '[0-9a-f]\{24\}' "$(dirname "$0")/.job_id${W:+_$W}.txt" | head -1)}
N=${1:-40}
SECS=${2:-12}
T=$(cat ~/.cache/huggingface/token)
curl -s -N --max-time "$SECS" -H "Authorization: Bearer $T" \
  "https://huggingface.co/api/jobs/Purusharth/$JOB/logs" \
  | python3 -c "
import sys, json
out = []
for line in sys.stdin:
    if line.startswith('data: '):
        try:
            d = json.loads(line[6:]); out.append(d['timestamp'][11:19] + ' ' + d['data'])
        except Exception:
            pass
print('\n'.join(out[-$N:]))"
