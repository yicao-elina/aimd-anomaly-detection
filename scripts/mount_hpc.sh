#!/bin/zsh

set -euo pipefail

REMOTE_USER="ycao73"
REMOTE_HOST="login.rockfish.jhu.edu"
REMOTE_PATH="/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp"
LOCAL_MOUNT="$HOME/hpc_mount"

mkdir -p "$LOCAL_MOUNT"

# Mount with reconnect and keepalive options for stability.
sshfs "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" "$LOCAL_MOUNT" \
  -o reconnect \
  -o compression=yes \
  -o defer_permissions \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3

echo "Mounted ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH} at ${LOCAL_MOUNT}"
