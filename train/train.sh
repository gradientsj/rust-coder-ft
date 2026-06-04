#!/usr/bin/env bash
# Launch a training run: train/train.sh configs/<config>.yaml [extra axolotl args]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate

CONFIG="${1:?usage: train/train.sh configs/<config>.yaml}"
shift || true

export AXOLOTL_DO_NOT_TRACK=1          # also works around a missing
                                       # telemetry/whitelist.yaml in the 0.16.1 wheel
export HF_HUB_ENABLE_HF_TRANSFER=1     # fast model/dataset downloads
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# NVLinked SXM box (Phase 0: all pairs NV18): NCCL auto-selects P2P/NVLS;
# no NCCL_P2P_DISABLE / IB tuning needed on a single node.

accelerate launch \
  --config_file "$REPO/train/fsdp_config.yaml" \
  -m axolotl.cli.train "$CONFIG" "$@"
