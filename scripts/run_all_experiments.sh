#!/usr/bin/env bash

for N in 10 15 20 25 30 35 40 45; do
  torchrun --standalone --nproc_per_node=2 -m src.training.train_triplet --config configs/train_classes_${N}.yaml
  python -m src.evaluation.evaluate    --config configs/eval_train_classes_${N}.yaml \
         --checkpoint runs/train_classes_${N}/best.pt --methods protonet knn cosine
done