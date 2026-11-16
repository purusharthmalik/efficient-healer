#!/usr/bin/env bash
# Queue one command on the GPU worker, with the current code, in a single commit.
#   infra/submit.sh 010_smoke infra/cmds/010_smoke.sh
set -euo pipefail
NAME=$1
CMD=$2
OPS_REPO=${OPS_REPO:-Purusharth/selfheal-ops}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/queue" "$STAGE/code"
cp "$CMD" "$STAGE/queue/$NAME.sh"
rsync -a --exclude '__pycache__' "$ROOT/selfheal" "$STAGE/code/"
mkdir -p "$STAGE/code/cmds" && cp "$ROOT"/infra/cmds/serve_*.sh "$STAGE/code/cmds/" 2>/dev/null || true

hf upload "$OPS_REPO" "$STAGE" . --repo-type dataset \
  --commit-message "queue $NAME" --quiet 2>&1 | grep -v -i "new version\|brew upgrade" || true
echo "queued $NAME"
