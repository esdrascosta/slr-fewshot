#!/usr/bin/env bash
# Distributed training launcher.
# Single node:  bash scripts/run_train_distributed.sh 4
# Multi node:   set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT then run on each node.
set -euo pipefail

NPROC="${1:-$(python -c 'import torch;print(max(torch.cuda.device_count(),1))')}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONFIG="${CONFIG:-configs/default.yaml}"

torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m src.training.train_triplet --config "${CONFIG}"
