import numpy as np
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from model.Image import Image
import os
import gc
FEATURE_KEYS = ("color", "texture", "hog", "shape", "venation")


def _to_1d_float32(vector_like) -> np.ndarray:
    if vector_like is None:
        return np.array([], dtype=np.float32)
    vector = np.asarray(vector_like, dtype=np.float32).flatten()
    if vector.size == 0:
        return np.array([], dtype=np.float32)
    return vector


def _fit_feature_stats(records):
    stats = {}
    for feature_key in FEATURE_KEYS:
        max_dim = max((item["raw_features"][feature_key].size for item in records), default=0)
        if max_dim == 0:
            stats[feature_key] = {
                "dim": 0,
                "mean": np.array([], dtype=np.float32),
                "std": np.array([], dtype=np.float32),
            }
            continue

        matrix = np.zeros((len(records), max_dim), dtype=np.float32)
        for row_index, item in enumerate(records):
            vector = item["raw_features"][feature_key]
            if vector.size == 0:
                continue
            limit = min(vector.size, max_dim)
            matrix[row_index, :limit] = vector[:limit]

        mean = matrix.mean(axis=0).astype(np.float32)
        std = matrix.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

        stats[feature_key] = {"dim": max_dim, "mean": mean, "std": std}
    return stats


def _normalize_vector(vector: np.ndarray, feature_stat: dict) -> np.ndarray:
    dim = feature_stat["dim"]
    if dim == 0:
        return np.array([], dtype=np.float32)

    padded = np.zeros(dim, dtype=np.float32)
    if vector.size:
        limit = min(vector.size, dim)
        padded[:limit] = vector[:limit]

    return ((padded - feature_stat["mean"]) / feature_stat["std"]).astype(np.float32)
