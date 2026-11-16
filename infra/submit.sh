#!/usr/bin/env bash
# Queue one command on a GPU worker, with the current code, in a single commit.
#   QUEUE=qa infra/submit.sh 010_smoke infra/cmds/010_smoke.sh
# The upload is verified and retried: a silent failure leaves a paid GPU idle.
set -uo pipefail
NAME=$1
CMD=$2
OPS_REPO=${OPS_REPO:-Purusharth/selfheal-ops}
QUEUE=${QUEUE:-queue}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/$QUEUE" "$STAGE/code/cmds"
cp "$CMD" "$STAGE/$QUEUE/$NAME.sh"
rsync -a --exclude '__pycache__' "$ROOT/selfheal" "$STAGE/code/"
cp "$ROOT"/infra/cmds/serve_*.sh "$ROOT"/infra/cmds/000_setup.sh "$STAGE/code/cmds/" 2>/dev/null || true

TOKEN=$(cat ~/.cache/huggingface/token)
for attempt in 1 2 3 4; do
  hf upload "$OPS_REPO" "$STAGE" . --repo-type dataset --commit-message "queue $NAME" --quiet > "$STAGE/.upload.log" 2>&1
  if curl -s -m 15 -H "Authorization: Bearer $TOKEN" "https://huggingface.co/api/datasets/$OPS_REPO/tree/main/$QUEUE" | grep -q "\"$QUEUE/$NAME.sh\""; then
    echo "queued $NAME in $QUEUE"
    exit 0
  fi
  echo "upload attempt $attempt failed: $(grep -v -i 'new version\|brew upgrade' "$STAGE/.upload.log" | tail -1 | cut -c1-200)"
  sleep $((attempt * 5))
done
echo "FAILED to queue $NAME in $QUEUE"
exit 1
