# Few-Shot Sign Language Recognition with a Contrastive Transformer

Official codebase for the paper *"A Transformer-Based Contrastive Learning
Approach for Few-Shot Sign Language Recognition"*
([arXiv:2204.02803](https://arxiv.org/abs/2204.02803)),
released by the authors many years after publication so the paper's results
can finally be reproduced end-to-end.

## Authors

- **Silvan Ferreira**\* — joint first author
- **Esdras Costa**\* — joint first author
- Márcio Dahia
- Jampierre Rocha

\* These authors contributed equally to this work.

A typo-checked version of the paper (LaTeX sources, figures, and PDF) is
available in [paper-files/](paper-files/); the original preprint is at
[arXiv:2204.02803](https://arxiv.org/abs/2204.02803).

Pipeline: LSA64 videos → MediaPipe Holistic keypoints → temporal interpolation →
Transformer encoder trained with triplet loss → few-shot evaluation
(Prototypical Networks / kNN / cosine similarity) + t-SNE visualization.

The code is **self-contained** (dataset download included) and supports
**distributed training** via `torchrun` (DDP). It runs on CPU, a single GPU, or
multiple GPUs/nodes without code changes.

## Notes vs. the paper

The paper leaves some details ambiguous; this repo makes them explicit and
configurable in `configs/default.yaml`:

| Ambiguity in the paper | Config key | Default |
|---|---|---|
| Input dim 225 (x, y, z) vs. 150 (z discarded) | `data.use_z` | `true` (225, matches Implementation Details) |
| Positional-encoding constant 1000 vs. 10000 | `model.pe_base` | `10000` (standard; set `1000` to match the paper's equations) |
| Sequence length after interpolation | `data.seq_len` | `64` |
| Triplet margin α | `train.margin` | `1.0` |
| Optimizer / LR / epochs / batch size | `train.*` | Adam, 1e-4, 30, 64 |
| Embedding head after encoder (Fig. 3 "Linear") | `model.proj_dim` | `128` |
| Train/test class split | `data.n_train_classes` | `48` (classes 1–48 train, 49–64 test) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Download LSA64 (~2 GB) and unpack to data/lsa64/videos
bash scripts/download_lsa64.sh data/lsa64

# 2) Extract MediaPipe Holistic keypoints -> data/lsa64/keypoints/*.npy
python -m src.data.extract_keypoints --videos data/lsa64/videos --out data/lsa64/keypoints

# 3) Train (single process)
python -m src.training.train_triplet --config configs/default.yaml

# 3') Train distributed (e.g. 4 GPUs on one node)
torchrun --standalone --nproc_per_node=4 -m src.training.train_triplet --config configs/default.yaml

# 3'') Multi-node example
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_ADDR:29500 \
         -m src.training.train_triplet --config configs/default.yaml

# 4) Evaluate (reproduces Tables 1-4 of the paper)
python -m src.evaluation.evaluate --config configs/default.yaml \
       --checkpoint runs/default/best.pt --methods protonet knn cosine

# 5) t-SNE plots (reproduces Figures 9-10)
python -m src.evaluation.tsne_viz --config configs/default.yaml \
       --checkpoint runs/default/best.pt --split test --out runs/default/tsne_test.png

# 6) Qualitative "dance vs. bathe" comparison (reproduces the comparison figure)
python -m src.evaluation.compare_signs --out runs/default/compare.png
#   add --skeleton-only to render from keypoints without the raw videos
```

To run the full paper pipeline end-to-end (training, sweep, evaluation, and
figures) in one go, use `bash scripts/run_all_experiments.sh`.

## Reproducing the class-diversity sweep (Tables 3-4)

Tables 3 and 4 vary the number of classes used to train the encoder (10 → 45 in
steps of 5) while always evaluating on the **same 16 held-out classes (49-64)**.
Because the train/test split is derived from `n_train_classes`, the sweep uses
two config families:

- `configs/train_classes_N.yaml` — trains the encoder on the first `N` classes;
- `configs/eval_train_classes_N.yaml` — keeps `n_train_classes: 48` so the test
  split stays fixed at classes 49-64, and points at the swept checkpoint.

```bash
for N in 10 15 20 25 30 35 40 45; do
  python -m src.training.train_triplet --config configs/train_classes_${N}.yaml
  python -m src.evaluation.evaluate    --config configs/eval_train_classes_${N}.yaml \
         --checkpoint runs/train_classes_${N}/best.pt --methods knn cosine
done
```

Each row of Tables 3-4 comes from `runs/eval_train_classes_N/results.json`; the
confusion matrices in Figures 7-8 are the `cm_knn.png` / `cm_cosine.png` written
to `runs/eval_train_classes_45/` (the best "45 train signs" setting).

## Repository layout

```
configs/default.yaml           # all hyperparameters
configs/*train_classes_*.yaml  # class-diversity sweep configs (Tables 3-4)
scripts/download_lsa64.sh      # dataset download + integrity check
scripts/run_all_experiments.sh # full paper pipeline in one command
scripts/run_train_distributed.sh # torchrun launcher helper
src/data/download.py           # python fallback downloader
src/data/extract_keypoints.py  # MediaPipe Holistic -> .npy per video
src/data/dataset.py            # keypoint datasets: triplet + episodic + flat
src/data/interpolation.py      # temporal length standardization
src/models/transformer.py      # pose embedding, PE, CLS token, encoder
src/training/train_triplet.py  # DDP training loop, checkpointing
src/evaluation/embeddings.py   # batch embedding extraction
src/evaluation/protocols.py    # protonet / kNN / cosine protocol implementations
src/evaluation/evaluate.py     # evaluation entry point (Tables 1-4)
src/evaluation/tsne_viz.py     # t-SNE figures
src/evaluation/compare_signs.py # qualitative "dance vs. bathe" figure
src/utils/distributed.py       # torchrun-aware init helpers
src/utils/misc.py              # seeding, config loading, logging
```

## Evaluation protocols (as in the paper)

- **ProtoNet, K-shot N-way** (Tables 1-2): episodes sampled from the 16 unseen
  test classes; prototypes = mean of support embeddings; squared Euclidean
  distance; accuracy averaged over `eval.n_episodes` episodes. Run with and
  without interpolation via `--no-interp` to reproduce the ablation.
- **kNN** (Table 3): for each of 40 repetitions, sample k support examples per
  unseen class as the reference set, then classify the remaining samples by
  majority vote among the k nearest neighbors (Euclidean distance), for k = 1..8.
- **Cosine** (Table 4): same reference sampling with n = 1..8 support samples;
  predict the class of the single most similar reference embedding.

All results are reported as **mean ± std** over repetitions (the paper reports
means only).

## LSA64 dataset

64 Argentinian Sign Language signs, 10 signers × 5 repetitions = 3200 videos.
Download URLs occasionally move; if `scripts/download_lsa64.sh` fails, get the
"raw" set from the maintainers' page (http://facundoq.github.io/datasets/lsa64/)
and unpack the .mp4/.avi files into `data/lsa64/videos/`. File names follow
`SSS_PPP_RRR.mp4` (sign_signer_repetition), which the code relies on for labels.

## Citation

If you use this code or build on the paper, please cite:

```bibtex
@misc{ferreira2022transformer,
  title         = {A Transformer-Based Contrastive Learning Approach for
                   Few-Shot Sign Language Recognition},
  author        = {Ferreira, Silvan and Costa, Esdras and Dahia, M{\'a}rcio
                   and Rocha, Jampierre},
  year          = {2022},
  eprint        = {2204.02803},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2204.02803}
}
```

## License

Research reproduction code, MIT. LSA64 has its own terms — see its page.
