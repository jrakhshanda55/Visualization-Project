# compute/projection.py
from __future__ import annotations
import numpy as np

from sklearn.manifold import TSNE
try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


def _standardize(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True) + 1e-8
    return (Z - mu) / sd


def project_2d(
    Z: np.ndarray,
    method: str = "tsne",
    standardize: bool = True,
    random_state: int = 0,
    **kwargs,
) -> np.ndarray:
    """
    Project embeddings Z -> 2D.

    method: "tsne" | "umap"
    standardize: if True, z-score before projection (recommended)
    random_state: seed for reproducibility
    kwargs: forwarded to the underlying projector
      - PCA: n_components is fixed to 2
      - TSNE: you may pass perplexity, learning_rate, etc.
      - UMAP: you may pass n_neighbors, min_dist, etc.
    """
    Z_ = _standardize(Z) if standardize else np.asarray(Z, dtype=float)

    m = method.lower()

    if m == "tsne":
        # set safe defaults if not provided
        perplexity = kwargs.pop("perplexity", max(5, min(30, (len(Z_) - 1) // 3)) )
        learning_rate = kwargs.pop("learning_rate", "auto")
        init = kwargs.pop("init", "pca")
        model = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            init=init,
            random_state=random_state,
            **kwargs
        )
        return model.fit_transform(Z_)

    if m == "umap":
        if not _HAS_UMAP:
            raise ImportError("UMAP not installed. Run: pip install umap-learn")
        n_neighbors = kwargs.pop("n_neighbors", 15)
        min_dist = kwargs.pop("min_dist", 0.1)
        metric = kwargs.pop("metric", "euclidean")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            **kwargs
        )
        return reducer.fit_transform(Z_)

    raise ValueError(f"Unknown projection method: {method}")
