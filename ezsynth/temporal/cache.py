"""Content-addressed, atomic flow cache. Never unpickles NumPy data."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

SCHEMA = 1


def frame_hash(frame: np.ndarray) -> str:
    array = np.ascontiguousarray(frame)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_flow(path, flow, source_hash, target_hash, producer):
    path = Path(path)
    field = np.asarray(flow, dtype=np.float32)
    if field.ndim != 3 or field.shape[-1] != 2 or not np.isfinite(field).all():
        raise ValueError("refusing to cache an invalid flow field")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, flow=field, schema=np.array(SCHEMA),
                                source_hash=np.array(source_hash), target_hash=np.array(target_hash),
                                producer=np.array(producer))
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_flow(path, shape, source_hash, target_hash, producer=None):
    with np.load(path, allow_pickle=False) as data:
        if int(data["schema"].item()) != SCHEMA:
            raise ValueError("flow cache schema mismatch")
        if str(data["source_hash"].item()) != source_hash or str(data["target_hash"].item()) != target_hash:
            raise ValueError("flow cache does not match the current frame content")
        if producer is not None and str(data["producer"].item()) != producer:
            raise ValueError("flow cache producer/checkpoint mismatch")
        flow = data["flow"]
        if flow.shape != shape or flow.dtype != np.float32 or not np.isfinite(flow).all():
            raise ValueError("flow cache shape, dtype, or values are invalid")
        return flow.copy()
