#!/usr/bin/env bash
# Environment bootstrap for rust-coder-ft on Lambda 4x H100.
#
# Order matters:
#   1. fresh venv; install torch 2.8.0+cu128 FIRST (axolotl 0.16.1 pins
#      torch==2.8.0; system torch is 2.7.0 which no axolotl release supports,
#      so the venv gets its own torch matching the cu12.8 toolchain).
#   2. main deps from pyproject (constrained so nothing replaces torch).
#   3. flash-attn + transformer-engine LAST with --no-build-isolation
#      (their builds import torch at metadata time).
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv

python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel

# Keep the resolver off torch: anything that tries to bump it fails loudly
# instead of silently pulling a different CUDA build.
echo "torch==2.8.0" > constraints.txt

pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install -c constraints.txt -e .

# Kernel libraries (prebuilt wheels for torch 2.8 + cu12.8 + py310 exist;
# falls back to source build against nvcc 12.8 if not).
pip install -c constraints.txt --no-build-isolation flash-attn==2.8.3

# transformer_engine_torch compiles from source and needs the cuDNN headers,
# which live in the pip nvidia-cudnn-cu12 package (NOT in system include paths
# on this image — without these vars the build dies on `cudnn.h: No such file`).
CUDNN=$PWD/$VENV/lib/python3.10/site-packages/nvidia/cudnn
CUDNN_PATH=$CUDNN CUDNN_HOME=$CUDNN CPLUS_INCLUDE_PATH=$CUDNN/include MAX_JOBS=32 \
  pip install -c constraints.txt --no-build-isolation "transformer_engine[pytorch]"

echo "=== import check ==="
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())
import flash_attn
print("flash_attn", flash_attn.__version__)
import transformer_engine.pytorch as te
import transformer_engine
print("transformer_engine", transformer_engine.__version__)
import axolotl, transformers, trl, datasets, datasketch, wandb
print("axolotl", axolotl.__version__)
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("datasets", datasets.__version__)
print("ALL IMPORTS OK")
EOF
