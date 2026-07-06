"""Reproduce the paper's Tables 1-4 and confusion matrices.

Usage:
    python -m src.evaluation.evaluate --config configs/default.yaml \
        --checkpoint runs/default/best.pt --methods protonet knn cosine \
        [--no-interp]   # interpolation ablation (Tables 1-2 "No" rows)

Results are printed and written to <run_dir>/results.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import KeypointDataset, train_test_classes
from src.evaluation.embeddings import extract_embeddings
from src.evaluation.protocols import (
    confusion,
    cosine_protocol,
    knn_protocol,
    protonet_episodes,
)
from src.models.transformer import build_model
from src.utils.misc import load_config, set_seed


def load_checkpoint(cfg: dict, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt.get("config", cfg), ckpt["feature_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--methods", nargs="+",
                    default=["protonet", "knn", "cosine"],
                    choices=["protonet", "knn", "cosine"])
    ap.add_argument("--no-interp", action="store_true",
                    help="disable temporal interpolation (ablation)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, e = cfg["data"], cfg["eval"]

    _, test_cls = train_test_classes(64, d["n_train_classes"])
    test_ds = KeypointDataset(
        d["keypoints_dir"], test_cls, d["seq_len"], d["use_z"],
        interpolate=d["interpolate"] and not args.no_interp,
    )
    model = load_checkpoint(cfg, args.checkpoint, device)
    emb, labels = extract_embeddings(model, test_ds, device)

    results: dict = {"interpolation": not args.no_interp}

    if "protonet" in args.methods:
        results["protonet"] = {}
        print("\n== Prototypical Networks (Tables 1-2) ==")
        for way in e["ways"]:
            for shot in e["shots"]:
                acc, ci = protonet_episodes(
                    emb, labels, way, shot, e["n_query"],
                    e["n_episodes"], seed=cfg["seed"],
                )
                results["protonet"][f"{way}way_{shot}shot"] = [acc, ci]
                print(f"{way:>2}-way {shot:>2}-shot: {acc:.3f} ± {ci:.3f} (95% CI)")

    if "knn" in args.methods:
        results["knn"] = {}
        print("\n== kNN (Table 3) ==")
        for k in e["knn_ks"]:
            acc, std = knn_protocol(emb, labels, k, e["n_repetitions"],
                                    seed=cfg["seed"])
            results["knn"][f"k{k}"] = [acc, std]
            print(f"k={k}: {acc:.3f} ± {std:.3f}")

    if "cosine" in args.methods:
        results["cosine"] = {}
        print("\n== Cosine similarity (Table 4) ==")
        for n in e["cos_ns"]:
            acc, std = cosine_protocol(emb, labels, n, e["n_repetitions"],
                                       seed=cfg["seed"])
            results["cosine"][f"n{n}"] = [acc, std]
            print(f"n={n}: {acc:.3f} ± {std:.3f}")

    # Confusion matrices at the best setting (n = k = 8), Figures 7-8.
    # Clamp support size so at least one query per class remains.
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    min_per_class = int(np.bincount(labels).max() and
                        min(np.bincount(labels)[np.unique(labels)]))
    n_support = max(1, min(8, min_per_class - 1))
    for method in ("knn", "cosine"):
        if method not in args.methods:
            continue
        cm, classes = confusion(emb, labels, n_support=n_support, method=method,
                                seed=cfg["seed"])
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ticks = [f"{c+1:03d}" for c in classes]
        ax.set_xticks(range(len(classes)), ticks, rotation=90, fontsize=7)
        ax.set_yticks(range(len(classes)), ticks, fontsize=7)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        out = run_dir / f"cm_{method}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"saved {out}")

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results -> {run_dir/'results.json'}")


if __name__ == "__main__":
    main()
