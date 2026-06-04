#!/usr/bin/env bash
# Persist irreplaceable artifacts to the HF Hub before instance shutdown.
#
#   export/upload_hub.sh <local-model-dir> <hub-repo-id> [--public]
# e.g.
#   export/upload_hub.sh export/qwen3-8b-ft   gradientsj/rust-coder-8b
#   export/upload_hub.sh export/qwen3-8b-ft-fp8 gradientsj/rust-coder-8b-fp8
#
# One-time auth on the instance:  .venv/bin/hf auth login
# Uploads run at datacenter uplink speed (minutes for 16GB), resumable.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
HF="$REPO/.venv/bin/hf"

SRC="${1:?usage: export/upload_hub.sh <local-dir> <user/repo> [--public]}"
DEST="${2:?usage: export/upload_hub.sh <local-dir> <user/repo> [--public]}"
VIS="--private"
[ "${3:-}" = "--public" ] && VIS=""

export HF_HUB_ENABLE_HF_TRANSFER=1
$HF repo create "$DEST" $VIS --exist-ok
$HF upload "$DEST" "$SRC" .
echo "uploaded $SRC -> https://huggingface.co/$DEST"
