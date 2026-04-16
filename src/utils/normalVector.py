import numpy as np
import os
from functools import lru_cache


def as_float_vector(feature) -> np.ndarray:
    if feature is None:
        return np.array([], dtype=np.float32)
    vector = np.asarray(feature, dtype=np.float32).flatten()
    return vector if vector.size else np.array([], dtype=np.float32)


@lru_cache(maxsize=2)
def _load_normalization_params_from_db(connection_string: str) -> dict:
    from dao.DAOPostgresql import DAOPostgresql

    dao = DAOPostgresql(connection_string)
    try:
        dao.connect()
        return dao.get_feature_normalization_params()
    finally:
        dao.close()


def get_normalization_params_from_db(connection_string: str | None = None) -> dict:
    conn_str = connection_string or os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql://admin:admin123@localhost:5432/mydb",
    )
    return _load_normalization_params_from_db(conn_str)

def _normalize_vector(vector: np.ndarray, feature_stat: dict) -> np.ndarray:
    dim = int(feature_stat.get("dim") or 0)
    if dim == 0:
        return np.array([], dtype=np.float32)

    padded = np.zeros(dim, dtype=np.float32)
    if vector.size:
        limit = min(vector.size, dim)
        padded[:limit] = vector[:limit]

    mean = as_float_vector(feature_stat.get("mean"))
    std = as_float_vector(feature_stat.get("std"))

    if mean.size < dim:
        mean = np.pad(mean, (0, dim - mean.size), mode="constant")
    else:
        mean = mean[:dim]

    if std.size < dim:
        std = np.pad(std, (0, dim - std.size), mode="constant", constant_values=1.0)
    else:
        std = std[:dim]

    std = np.where(std < 1e-8, 1.0, std)
    return ((padded - mean) / std).astype(np.float32)


def normalize_vector_by_feature_name(
    vector,
    feature_name: str,
    normalization_params: dict | None = None,
    connection_string: str | None = None,
) -> np.ndarray:
    params = normalization_params if normalization_params is not None else get_normalization_params_from_db(connection_string)
    feature_stat = params.get(feature_name)
    if not feature_stat:
        return as_float_vector(vector)
    return _normalize_vector(as_float_vector(vector), feature_stat)