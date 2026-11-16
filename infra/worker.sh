#!/usr/bin/env bash
# GPU worker. Runs inside one long HF Job and executes queued shell commands,
# so the Python environments and model cache stay warm between phases.
#
# Protocol (all in the private ops dataset repo $OPS_REPO):
#   queue/NNN_name.sh   command to run, executed once, in name order
#   code/...            the project code, synced before every command
#   logs/NNN_name.log   full output, uploaded when the command ends
#   logs/NNN_name.rc    its exit code
#   queue/STOP          tells the worker to exit (stops billing)
set -u
W=/work
mkdir -p "$W/sync" "$W/logs" "$W/out"
cd "$W"
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DISABLE_PROGRESS_BARS=1

log() { echo "[worker $(date -u +%H:%M:%S)] $*"; }

uv tool install -q "huggingface_hub[cli]" || { log "hf cli install failed"; exit 1; }
log "started; ops repo $OPS_REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || log "no GPU visible"
df -h "$W" | tail -1

while true; do
  hf download "$OPS_REPO" --repo-type dataset --include "queue/*" --include "code/**" \
    --local-dir "$W/sync" --quiet >/dev/null 2>&1 || log "sync failed, retrying"

  for f in $(ls "$W"/sync/queue/*.sh 2>/dev/null | sort); do
    n=$(basename "$f" .sh)
    [ -f "$W/logs/$n.rc" ] && continue
    log "START $n"
    ( cd "$W" && bash "$f" ) 2>&1 | tee "$W/logs/$n.log"
    rc=${PIPESTATUS[0]}
    echo "$rc" > "$W/logs/$n.rc"
    log "END $n rc=$rc"
    hf upload "$OPS_REPO" "$W/logs" logs --repo-type dataset --include "$n.*" \
      --commit-message "log $n rc=$rc" --quiet >/dev/null 2>&1 || log "log upload failed"
    # Results too, so they can be inspected locally and survive a job restart.
    hf upload "$OPS_REPO" "$W/out" out --repo-type dataset --exclude "*.safetensors" --exclude "*.bin" \
      --commit-message "out after $n" --quiet >/dev/null 2>&1 || log "out upload failed"
    # Re-sync before the next command so a fix pushed meanwhile is picked up.
    break
  done

  if [ -f "$W/sync/queue/STOP" ]; then log "STOP found, exiting"; exit 0; fi
  sleep 6
done
