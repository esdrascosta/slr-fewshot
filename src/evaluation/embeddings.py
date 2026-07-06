from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import KeypointDataset


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataset: KeypointDataset,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings (M, D), labels (M,)) for the whole dataset."""
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=num_workers, shuffle=False)
    embs, labels = [], []
    for x, y in loader:
        embs.append(model(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
    if was_training:
        model.train()
    return np.concatenate(embs), np.concatenate(labels)
