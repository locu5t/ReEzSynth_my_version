"""Pairwise flow with real reverse inference and per-content cache validation."""
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import cv2
import numpy as np

from .cache import file_hash, frame_hash, load_flow, save_flow


class PairFlow:
    def __init__(self, frames, config, project, precomputation):
        self.frames = frames
        self.config = config
        self.project = project
        self.precomputation = precomputation
        self.hashes = [frame_hash(frame) for frame in frames]
        self.engine = None
        self.read_only = False
        signature = {"implementation": file_hash(__file__), "backend": config.flow_backend,
                     "opencv": cv2.__version__, "schema": 1}
        if config.flow_backend == "legacy":
            name = precomputation.flow_engine.upper()
            model = precomputation.flow_model
            if name == "RAFT":
                model_path = Path("models") / "raft" / f"raft-{model}.pth"
            elif name == "NEUFLOW":
                model_path = Path("models") / "neuflow" / f"{model}.pth"
            else:
                raise ValueError(f"Unsupported legacy flow engine: {name}")
            if not model_path.is_file():
                raise FileNotFoundError(f"Flow checkpoint not found: {model_path.resolve()}")
            signature.update(engine=name, model=model, checkpoint_sha256=file_hash(model_path))
        self.producer = json.dumps(signature, sort_keys=True)
        namespace = hashlib.sha256(self.producer.encode()).hexdigest()[:20]
        self.directory = Path(project.cache_dir) / "temporal_flow_v1" / namespace
        if config.flow_backend == "precomputed":
            if not config.precomputed_flow_dir:
                raise ValueError("precomputed flow requires precomputed_flow_dir")
            self.directory = Path(config.precomputed_flow_dir)

    def _path(self, source, target):
        suffix = "" if self.config.flow_backend == "precomputed" else f"_{self.hashes[source][:12]}_{self.hashes[target][:12]}"
        return self.directory / f"{source:05d}_to_{target:05d}{suffix}.npz"

    def _compute(self, source, target):
        if self.config.flow_backend == "opencv":
            if self.engine is None:
                self.engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            a = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            b = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            if min(a.shape) < 12:
                raise ValueError("DIS flow requires images at least 12 pixels on each side")
            return self.engine.calc(a, b, None)
        if self.engine is None:
            from ..engines.flow_engine import NeuFlowEngine, RAFTFlowEngine
            if self.precomputation.flow_engine.upper() == "RAFT":
                self.engine = RAFTFlowEngine(self.precomputation.flow_model)
            else:
                self.engine = NeuFlowEngine(self.precomputation.flow_model)
        # The bundled wrappers only transpose tensors. The pretrained networks
        # require RGB; OpenCV-loaded project frames are BGR.
        rgb = [np.ascontiguousarray(frame[..., ::-1]) for frame in (source, target)]
        return self.engine.compute(rgb)[0]

    def get(self, source, target):
        if source == target:
            return np.zeros(self.frames[source].shape[:2] + (2,), np.float32)
        path = self._path(source, target)
        shape = self.frames[source].shape[:2] + (2,)
        external = self.config.flow_backend == "precomputed"
        force = self.project.force_recompute_flow and not self.read_only and not external
        if not force:
            try:
                return load_flow(path, shape, self.hashes[source], self.hashes[target],
                                 None if external else self.producer)
            except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
                if external or self.read_only:
                    raise ValueError(f"Required flow {path} is missing or invalid: {exc}") from exc
        flow = np.ascontiguousarray(self._compute(self.frames[source], self.frames[target]),
                                    dtype=np.float32)
        if flow.shape != shape or not np.isfinite(flow).all():
            raise ValueError("flow backend returned an invalid field")
        save_flow(path, flow, self.hashes[source], self.hashes[target], self.producer)
        return flow

    def prepare(self, pairs):
        try:
            for source, target in sorted(set(pairs)):
                self.get(source, target)
        finally:
            self.engine = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.read_only = True
