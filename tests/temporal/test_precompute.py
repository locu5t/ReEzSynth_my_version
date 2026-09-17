"""Adapter contract tests use synthetic model doubles; no pretrained claims."""
import argparse
import json
import sys
import types

import numpy as np
import pytest
import torch

from ezsynth.temporal import precompute as worker
from ezsynth.temporal.cache import frame_hash, load_flow
from ezsynth.temporal.config import TemporalConfig


def test_missing_face_or_blink_score_is_unknown():
    result = types.SimpleNamespace(face_landmarks=[], face_blendshapes=[])
    assert worker.face_record(result, 16, 16, None, "subject") is None
    result = types.SimpleNamespace(face_landmarks=[[types.SimpleNamespace(x=0.5, y=0.5)]],
                                   face_blendshapes=[[]])
    output = worker.face_record(result, 16, 16, None, "subject")
    assert output["left"] is None and output["right"] is None


def test_closed_eye_and_anatomical_side_are_retained():
    edge = types.SimpleNamespace(start=0, end=1)
    connections = types.SimpleNamespace(FACE_LANDMARKS_LEFT_EYE=[edge], FACE_LANDMARKS_RIGHT_EYE=[edge])
    result = types.SimpleNamespace(
        face_landmarks=[[types.SimpleNamespace(x=0.2, y=0.5), types.SimpleNamespace(x=0.7, y=0.5)]],
        face_blendshapes=[[types.SimpleNamespace(category_name="eyeBlinkLeft", score=1.0),
                          types.SimpleNamespace(category_name="eyeBlinkRight", score=0.0)]])
    output = worker.face_record(result, 20, 20, connections, "subject")
    assert output["left"]["openness"] == 0
    assert output["right"]["openness"] == 1
    assert len(output["left"]["polygon"]) >= 3


def test_revision_mismatch_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="wrong"))
    with pytest.raises(ValueError, match="must be checked out"):
        worker.verified_repository(tmp_path, "waft")


def test_cotracker_requires_noncommercial_acknowledgement():
    with pytest.raises(ValueError, match="CC-BY-NC"):
        worker.cotracker_job(types.SimpleNamespace(acknowledge_noncommercial=False), None, None, None, None)


def test_cotracker_rgb_coordinates_and_shot_reset_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "verified_repository", lambda *a: tmp_path)
    seen = []
    class Model(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            assert kwargs["offline"] and not kwargs["v2"]
        def forward(self, video, queries, backward_tracking):
            assert backward_tracking
            np.testing.assert_array_equal(video[0, 0, :, 0, 0], [30, 20, 10])
            seen.append(video.shape)
            tracks = queries[..., 1:][:, None].expand(1, video.shape[1], -1, -1).clone()
            return tracks, torch.ones(tracks.shape[:-1], dtype=torch.bool)
    module = types.ModuleType("cotracker.predictor")
    module.CoTrackerPredictor = Model
    monkeypatch.setitem(sys.modules, "cotracker.predictor", module)
    model_path = tmp_path / "dummy.pth"
    model_path.write_bytes(b"dummy")
    args = argparse.Namespace(acknowledge_noncommercial=True, repository=str(tmp_path),
                              model=str(model_path), device="cpu", max_side=8,
                              max_frames=10, grid_size=2, output=str(tmp_path / "tracks.npz"))
    frame = np.empty((16, 20, 3), np.uint8)
    frame[:] = [10, 20, 30]
    frames = [frame] * 4
    config = types.SimpleNamespace(project=types.SimpleNamespace(style_indices=[0, 2]))
    worker.cotracker_job(args, config, frames, [frame_hash(f) for f in frames], np.array([0, 0, 1, 1]))
    with np.load(args.output, allow_pickle=False) as bundle:
        assert bundle["tracks"].shape == (4, 8, 2)
        assert not bundle["visibility"][:2, 4:].any()
        assert not bundle["visibility"][2:, :4].any()
        assert bundle["tracks"][0, :4, 0].max() == 19
    assert len(seen) == 2


def test_waft_cache_export_contract_with_fake_model(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "verified_repository", lambda *a: tmp_path)
    timm = types.ModuleType("timm")
    timm.create_model = lambda *a, **k: None
    model_module = types.ModuleType("model")
    a1 = types.ModuleType("model.waft_a1")
    a1.DepthAnythingFeature = lambda **k: None
    model_module.waft_a1 = a1
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
    model_module.fetch_model = lambda args: Model()
    inference = types.ModuleType("inference_tools")
    calls = []
    class Wrapper:
        def __init__(self, model, **kwargs):
            assert not kwargs["tiling"]
        def calc_flow(self, a, b):
            calls.append((a.shape, b.shape))
            return {"flow": [torch.zeros((1, 2, a.shape[2], a.shape[3]))]}
    inference.InferenceWrapper = Wrapper
    for name, module in (("timm", timm), ("model", model_module),
                         ("model.waft_a1", a1), ("inference_tools", inference)):
        monkeypatch.setitem(sys.modules, name, module)
    checkpoint = tmp_path / "model.pth"
    torch.save(Model().state_dict(), checkpoint)
    settings = tmp_path / "waft.json"
    settings.write_text(json.dumps({"algorithm": "waft-a1", "image_size": [16, 16]}))
    args = argparse.Namespace(repository=str(tmp_path), model=str(checkpoint), waft_config=str(settings),
                              max_pairs=10, device="cpu", output=str(tmp_path / "flows"))
    frame = np.zeros((16, 16, 3), np.uint8)
    config = types.SimpleNamespace(project=types.SimpleNamespace(style_indices=[0]), temporal=TemporalConfig())
    hashes = [frame_hash(frame)] * 2
    worker.waft_job(args, config, [frame, frame], hashes, np.array([0, 0]))
    assert len(calls) == 2  # both directions, never a sign flip
    for a, b in ((0, 1), (1, 0)):
        flow = load_flow(tmp_path / "flows" / f"{a:05d}_to_{b:05d}.npz", (16, 16, 2), hashes[a], hashes[b])
        assert not flow.any()
    worker.waft_job(args, config, [frame, frame], hashes, np.array([0, 0]))
    assert len(calls) == 2  # repeat uses validated cache
