"""
Clustering utilities using UMAP + HDBSCAN.
"""

import logging
import numpy as np
import faiss
import umap
import hdbscan
from typing import List, Dict

from .models import Chunk
from .config import Config

log = logging.getLogger("Refinery")


def clusterize(chunks: List[Chunk], cfg: Config) -> Dict[int, int]:
    ccfg = cfg.cluster_cfg(-1)
    log.info("[UMAP] Reducing to %d dims...", ccfg["umap_components"])
    mat = np.vstack([c.embedding for c in chunks])
    reducer = umap.UMAP(
        n_components=ccfg["umap_components"],
        n_neighbors=min(ccfg["umap_neighbors"], len(chunks)-1),
        min_dist=ccfg["umap_min_dist"],
        metric="cosine", random_state=42
    )
    coords = reducer.fit_transform(mat)
    for c, co in zip(chunks, coords):
        c.umap_coords = co

    log.info("[HDBSCAN] Clustering (min_cluster=%d, min_samples=%d)...",
             ccfg["hdbscan_min_cluster"], ccfg["hdbscan_min_samples"])
    labels = hdbscan.HDBSCAN(
        min_cluster_size=ccfg["hdbscan_min_cluster"],
        min_samples=ccfg["hdbscan_min_samples"],
        metric="euclidean", cluster_selection_method="eom"
    ).fit_predict(np.vstack([c.umap_coords for c in chunks]))

    stats = {}
    for c, lab in zip(chunks, labels):
        c.cluster_id = int(lab)
        stats[int(lab)] = stats.get(int(lab), 0) + 1
    valid = {k: v for k, v in stats.items() if k != -1}
    log.info("[HDBSCAN] %d clusters, %d noise", len(valid), stats.get(-1, 0))
    return stats


def medoids(chunks: List[Chunk], samples_per_cluster: int) -> Dict[int, List[Chunk]]:
    out = {}
    for cid in sorted(set(c.cluster_id for c in chunks if c.cluster_id != -1)):
        members = [c for c in chunks if c.cluster_id == cid]
        coords = np.vstack([c.umap_coords for c in members]).astype("float32")
        idx = faiss.IndexFlatL2(coords.shape[1])
        idx.add(coords)
        _, ids = idx.search(coords.mean(axis=0, keepdims=True), min(samples_per_cluster, len(members)))
        out[cid] = [members[i] for i in ids[0]]
    return out