"""torchrun-aware distributed helpers.

Works transparently in three modes:
  - plain `python -m ...`      -> single process, no process group
  - `torchrun --nproc_per_node=k` on 1 node
  - `torchrun --nnodes=m ...`  multi-node
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup() -> tuple[int, int, torch.device]:
    """Initialize (if launched by torchrun) and return (rank, world, device)."""
    if is_distributed():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        if torch.cuda.is_available():
            local = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local)
            device = torch.device("cuda", local)
        else:
            device = torch.device("cpu")
        return rank, world, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        t = t.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t


@contextmanager
def main_process_first(rank: int):
    """Let rank 0 do a cached step (e.g. dataset scan) before others."""
    if rank != 0:
        barrier()
    yield
    if rank == 0:
        barrier()
