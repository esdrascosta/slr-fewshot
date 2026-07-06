"""t-SNE visualization of the embedding space (paper Figures 9-10).

Usage:
    python -m src.evaluation.tsne_viz --config configs/default.yaml \
        --checkpoint runs/default/best.pt --split test --out tsne_test.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from src.data.dataset import KeypointDataset, train_test_classes
from src.evaluation.embeddings import extract_embeddings
from src.evaluation.evaluate import load_checkpoint
from src.utils.misc import load_config, set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--out", default="tsne.png")
    ap.add_argument("--perplexity", type=float, default=30.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = cfg["data"]

    train_cls, test_cls = train_test_classes(64, d["n_train_classes"])
    classes = train_cls if args.split == "train" else test_cls
    ds = KeypointDataset(d["keypoints_dir"], classes, d["seq_len"],
                         d["use_z"], d["interpolate"])
    model = load_checkpoint(cfg, args.checkpoint, device)
    emb, labels = extract_embeddings(model, ds, device)

    xy = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
              random_state=cfg["seed"]).fit_transform(emb)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="hsv", s=8, alpha=0.8)
    ax.set_xlabel("x_emb")
    ax.set_ylabel("y_emb")
    ax.set_title(f"t-SNE of {args.split} embeddings "
                 f"({len(set(labels.tolist()))} classes)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
