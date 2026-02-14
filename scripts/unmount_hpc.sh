#!/bin/zsh

set -euo pipefail

LOCAL_MOUNT="$HOME/hpc_mount"

if mount | grep -q "$LOCAL_MOUNT"; then
  umount "$LOCAL_MOUNT"
  echo "Unmounted ${LOCAL_MOUNT}"
else
  echo "${LOCAL_MOUNT} is not mounted"
fi
