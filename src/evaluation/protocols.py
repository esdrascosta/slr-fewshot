"""Few-shot classification protocols (paper Section 5), on precomputed
embeddings of the *unseen* test classes.

- protonet_episodes: K-shot N-way episodes; prototypes = support means;
  squared Euclidean distance (Eqs. 14-16).
- knn_protocol: sample k support examples per class as the reference set,
  classify the remaining samples by majority vote among the k nearest
  (Euclidean) neighbors. As in the paper, k == n (support size).
- cosine_protocol: sample n support examples per class; predict the class of
  the single most similar reference by cosine similarity.

All return (mean_accuracy, dispersion) over episodes/repetitions.
"""
from __future__ import annotations

import numpy as np


def _split_support_query(
    rng: np.random.Generator,
    embeddings: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    n_support: int,
    n_query: int | None = None,
):
    sup_idx, qry_idx = [], []
    for c in classes:
        idx = rng.permutation(np.where(labels == c)[0])
        sup_idx.append(idx[:n_support])
        qry = idx[n_support:] if n_query is None else idx[n_support:n_support + n_query]
        qry_idx.append(qry)
    sup_idx = np.concatenate(sup_idx)
    qry_idx = np.concatenate(qry_idx)
    return (
        embeddings[sup_idx], labels[sup_idx],
        embeddings[qry_idx], labels[qry_idx],
    )


def protonet_episodes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_way: int,
    k_shot: int,
    n_query: int,
    n_episodes: int,
    seed: int = 0,
) -> tuple[float, float]:
    """Mean accuracy and 95% CI half-width over episodes."""
    rng = np.random.default_rng(seed)
    all_classes = np.unique(labels)
    accs = []
    for _ in range(n_episodes):
        classes = rng.choice(all_classes, size=n_way, replace=False)
        xs, ys, xq, yq = _split_support_query(
            rng, embeddings, labels, classes, k_shot, n_query
        )
        protos = np.stack([xs[ys == c].mean(axis=0) for c in classes])  # (N, D)
        d2 = ((xq[:, None, :] - protos[None, :, :]) ** 2).sum(-1)       # (Q, N)
        pred = classes[d2.argmin(axis=1)]
        accs.append((pred == yq).mean())
    accs = np.asarray(accs)
    ci95 = 1.96 * accs.std(ddof=1) / np.sqrt(len(accs))
    return float(accs.mean()), float(ci95)


def knn_protocol(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
    n_repetitions: int = 40,
    seed: int = 0,
) -> tuple[float, float]:
    """Mean accuracy ± std over repetitions; support size per class == k."""
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    accs = []
    for _ in range(n_repetitions):
        xs, ys, xq, yq = _split_support_query(rng, embeddings, labels, classes, k)
        clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        clf.fit(xs, ys)
        accs.append(float((clf.predict(xq) == yq).mean()))
    accs = np.asarray(accs)
    return float(accs.mean()), float(accs.std(ddof=1))


def cosine_protocol(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n: int,
    n_repetitions: int = 40,
    seed: int = 0,
) -> tuple[float, float]:
    """1-NN by cosine similarity over n support samples per class."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    accs = []
    for _ in range(n_repetitions):
        xs, ys, xq, yq = _split_support_query(rng, embeddings, labels, classes, n)
        xs_n = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-12)
        xq_n = xq / (np.linalg.norm(xq, axis=1, keepdims=True) + 1e-12)
        sims = xq_n @ xs_n.T                      # (Q, S)
        pred = ys[sims.argmax(axis=1)]
        accs.append(float((pred == yq).mean()))
    accs = np.asarray(accs)
    return float(accs.mean()), float(accs.std(ddof=1))


def confusion(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_support: int,
    method: str = "cosine",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Confusion matrix for one repetition (paper Figures 7-8)."""
    from sklearn.metrics import confusion_matrix
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    xs, ys, xq, yq = _split_support_query(rng, embeddings, labels, classes, n_support)
    if method == "cosine":
        xs_n = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-12)
        xq_n = xq / (np.linalg.norm(xq, axis=1, keepdims=True) + 1e-12)
        pred = ys[(xq_n @ xs_n.T).argmax(axis=1)]
    else:
        clf = KNeighborsClassifier(n_neighbors=n_support, metric="euclidean")
        clf.fit(xs, ys)
        pred = clf.predict(xq)
    return confusion_matrix(yq, pred, labels=classes), classes
