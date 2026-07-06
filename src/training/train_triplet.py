"""Train the Sign Transformer with triplet loss (paper Section 4).

Single process:
    python -m src.training.train_triplet --config configs/default.yaml
Distributed (DDP):
    torchrun --standalone --nproc_per_node=4 -m src.training.train_triplet \
        --config configs/default.yaml

Validation: quick 5-way 5-shot Prototypical episodes on the *unseen* test
classes (rank 0 only); best checkpoint by validation accuracy is kept.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.data.dataset import KeypointDataset, TripletDataset, train_test_classes
from src.evaluation.embeddings import extract_embeddings
from src.evaluation.protocols import protonet_episodes
from src.models.transformer import build_model
from src.utils import distributed as du
from src.utils.misc import load_config, log, set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rank, world, device = du.setup()
    set_seed(cfg["seed"], rank)
    run_dir = Path(cfg["run_dir"])
    if du.is_main(rank):
        run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    d = cfg["data"]
    train_cls, test_cls = train_test_classes(64, d["n_train_classes"])
    with du.main_process_first(rank):
        train_base = KeypointDataset(
            d["keypoints_dir"], train_cls, d["seq_len"], d["use_z"], d["interpolate"]
        )
        val_base = KeypointDataset(
            d["keypoints_dir"], test_cls, d["seq_len"], d["use_z"], d["interpolate"]
        )
    triplets = TripletDataset(train_base, d["n_triplets"], seed=cfg["seed"])
    sampler = (
        DistributedSampler(triplets, shuffle=True) if du.is_distributed() else None
    )
    loader = DataLoader(
        triplets,
        batch_size=cfg["train"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=d["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # ---------------- model ----------------
    model = build_model(cfg, train_base.feature_dim).to(device)
    if du.is_distributed():
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    core = model.module if isinstance(model, DDP) else model

    t = cfg["train"]
    opt = torch.optim.Adam(
        model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"])
    )
    criterion = torch.nn.TripletMarginLoss(margin=float(t["margin"]), p=2)
    use_amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_acc = 0, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        core.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch, best_acc = ckpt["epoch"] + 1, ckpt.get("best_acc", 0.0)
        log(rank, f"Resumed from {args.resume} at epoch {start_epoch}")

    # ---------------- loop ----------------
    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        triplets.set_epoch(epoch)  # resample triplets (identical on all ranks)
        if sampler is not None:
            sampler.set_epoch(epoch)  # ranks then draw disjoint subsets

        t0, running = time.time(), 0.0
        for step, (a, p, n) in enumerate(loader):
            a, p, n = (x.to(device, non_blocking=True) for x in (a, p, n))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                # single forward on the concatenated batch
                z = model(torch.cat([a, p, n], dim=0))
                za, zp, zn = z.chunk(3, dim=0)
                loss = criterion(za, zp, zn)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            if step % t["log_every"] == 0:
                log(rank, f"epoch {epoch} step {step:4d} loss {loss.item():.4f}")

        avg = du.reduce_mean(torch.tensor(running / len(loader), device=device))
        log(rank, f"epoch {epoch} done in {time.time()-t0:.1f}s  mean loss {avg:.4f}")

        # ------------- validation (rank 0) -------------
        if du.is_main(rank):
            emb, labels = extract_embeddings(core, val_base, device,
                                             batch_size=128)
            acc, ci = protonet_episodes(
                emb, labels, n_way=5, k_shot=5, n_query=5,
                n_episodes=t["val_episodes"], seed=cfg["seed"] + epoch,
            )
            log(rank, f"epoch {epoch} val 5-way 5-shot acc {acc:.3f} ± {ci:.3f}")
            state = {
                "model": core.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
                "best_acc": max(best_acc, acc),
                "config": cfg,
                "feature_dim": train_base.feature_dim,
            }
            torch.save(state, run_dir / "last.pt")
            if acc > best_acc:
                best_acc = acc
                torch.save(state, run_dir / "best.pt")
                log(rank, f"  new best ({best_acc:.3f}) -> {run_dir/'best.pt'}")
        du.barrier()

    du.cleanup()


if __name__ == "__main__":
    main()
