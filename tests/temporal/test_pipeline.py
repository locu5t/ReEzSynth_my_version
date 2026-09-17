"""CPU integration tests use explicit synthesis/flow doubles, not GPU claims."""
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from ezsynth.config import MainConfig
from ezsynth.temporal.cache import frame_hash
from ezsynth.temporal.pipeline import TemporalPipeline


class Data:
    def __init__(self, frames, styles, masks=None, modulations=None):
        self.frames, self.styles = frames, styles
        self.masks, self.modulations = masks, modulations

    def get_content_frames(self):
        return self.frames

    def get_style_frames(self):
        return self.styles

    def get_mask_frames(self):
        return self.masks

    def get_modulation_frames(self):
        return self.modulations


class ZeroFlow:
    def __init__(self, frames, *args):
        self.shape = frames[0].shape[:2] + (2,)

    def prepare(self, pairs):
        self.pairs = set(pairs)
        assert all((b, a) in self.pairs for a, b in self.pairs)

    def get(self, source, target):
        assert (source, target) in self.pairs
        return np.zeros(self.shape, np.float32)


class NnfSamplingEngine:
    def __init__(self):
        self.calls = []

    def run(self, style, guides, modulation_map, initial_nnf, output_nnf):
        assert not output_nnf
        assert initial_nnf.dtype == np.int32
        channels = sum(src.shape[2] for src, _, _ in guides)
        assert modulation_map.shape == style.shape[:2] + (channels,)
        assert modulation_map.dtype == np.uint8
        for src, dst, _ in guides:
            assert src.dtype == dst.dtype == np.uint8
            assert src.shape == dst.shape
        self.calls.append((style.copy(), guides, modulation_map.copy()))
        return style[initial_nnf[..., 1], initial_nnf[..., 0]], np.zeros(style.shape[:2], np.float32)


class CpuTestPipeline(TemporalPipeline):
    def _edges(self, frames):
        return [np.zeros_like(frame) for frame in frames]


def config(tmp_path, indices=(0,), **temporal):
    return MainConfig(project={"content_dir": "unused", "style_path": ["unused"] * len(indices),
                               "style_indices": list(indices), "output_dir": str(tmp_path),
                               "cache_dir": str(tmp_path / "cache")},
                      precomputation={}, pipeline={}, ebsynth_params={},
                      temporal={"enabled": True, "warp_device": "cpu", **temporal})


def test_multi_reference_output_length_order_and_exact_keyframes(tmp_path):
    frames = [np.full((16, 16, 3), 100, np.uint8)] * 4
    styles = [np.full_like(frames[0], 200), np.full_like(frames[0], 20)]
    engine = NnfSamplingEngine()
    pipeline = CpuTestPipeline(config(tmp_path, (3, 0)), Data(frames, styles), engine, ZeroFlow)
    out = pipeline.run()
    assert len(out) == 4
    np.testing.assert_array_equal(out[0], styles[1])
    np.testing.assert_array_equal(out[3], styles[0])
    assert len(engine.calls) == 4  # two references for each non-key frame
    report = json.loads((tmp_path / "temporal_report.json").read_text())
    assert [entry["frame"] for entry in report["frames"]] == [0, 1, 2, 3]
    assert report["confidence_is_calibrated"] is False


def test_all_keyframes_need_no_flow_or_edge_models(tmp_path):
    frame = np.zeros((16, 16, 3), np.uint8)
    pipeline = CpuTestPipeline(config(tmp_path, (0,)), Data([frame], [frame + 20]),
                               NnfSamplingEngine(), lambda *a: pytest.fail("must not load flow"))
    assert np.array_equal(pipeline.run()[0], frame + 20)


def test_one_reference_propagates_in_both_time_directions(tmp_path):
    frame = np.full((16, 16, 3), 30, np.uint8)
    engine = NnfSamplingEngine()
    pipeline = CpuTestPipeline(config(tmp_path, (1,)), Data([frame] * 3, [frame + 5]), engine, ZeroFlow)
    assert all(np.array_equal(out, frame + 5) for out in pipeline.run())
    assert len(engine.calls) == 2


def test_project_masks_preserve_unselected_content(tmp_path):
    frame = np.zeros((16, 16, 3), np.uint8)
    mask = np.zeros((16, 16), np.uint8)
    mask[:, :8] = 255
    pipeline = CpuTestPipeline(config(tmp_path), Data([frame] * 2, [frame + 100], [mask] * 2),
                               NnfSamplingEngine(), ZeroFlow)
    for output in pipeline.run():
        assert (output[:, :8] == 100).all() and not output[:, 8:].any()


def test_missing_keyframe_in_shot_fails_before_rendering(tmp_path):
    frame = np.zeros((16, 16, 3), np.uint8)
    engine = NnfSamplingEngine()
    pipeline = CpuTestPipeline(config(tmp_path, scene_cuts=[1]), Data([frame] * 2, [frame]), engine, ZeroFlow)
    with pytest.raises(ValueError, match="no painted keyframe"):
        pipeline.run()
    assert not engine.calls


@pytest.mark.parametrize("indices", [(0, 0), (-1,), (9,)])
def test_bad_keyframe_indices_fail_early(tmp_path, indices):
    frame = np.zeros((16, 16, 3), np.uint8)
    pipeline = CpuTestPipeline(config(tmp_path, indices), Data([frame] * 2, [frame] * len(indices)),
                               NnfSamplingEngine(), ZeroFlow)
    with pytest.raises(ValueError, match="style_indices"):
        pipeline.run()


def test_size_mismatch_is_not_silently_resized_twice(tmp_path):
    frame = np.zeros((16, 16, 3), np.uint8)
    pipeline = CpuTestPipeline(config(tmp_path), Data([frame], [frame[:8]]), NnfSamplingEngine(), ZeroFlow)
    with pytest.raises(ValueError, match="equal-resolution"):
        pipeline.run()


def test_missing_closed_eye_reference_is_reported_and_gated(tmp_path):
    frame = np.full((16, 16, 3), 80, np.uint8)
    def face(openness):
        return {"face_id": "actor", "left": {"openness": openness,
                "polygon": [[3, 4], [8, 4], [8, 6], [3, 6]]}, "right": None}
    path = tmp_path / "faces.json"
    path.write_text(json.dumps({"schema": 1, "frame_hashes": [frame_hash(frame)] * 2,
                                "frames": [face(1), face(0)]}))
    engine = NnfSamplingEngine()
    pipeline = CpuTestPipeline(config(tmp_path, face_annotations=str(path)),
                               Data([frame] * 2, [frame + 20]), engine, ZeroFlow)
    with pytest.warns(RuntimeWarning, match="insufficient reference"):
        pipeline.run()
    report = json.loads((tmp_path / "temporal_report.json").read_text())
    assert report["recommendations"][0]["unsupported_eyes"] == ["left"]
    modulation = engine.calls[0][2]
    assert not modulation[5, 5, 6:12].any()  # POSITION and WARP guide channels disabled
    assert modulation[5, 5, :6].min() == 255  # source RGB/edges still usable


def test_strict_missing_reference_mode_writes_report_before_error(tmp_path):
    a = np.zeros((16, 16, 3), np.uint8)
    b = np.full_like(a, 255)
    pipeline = CpuTestPipeline(config(tmp_path, missing_reference="error"), Data([a, b], [a]),
                               NnfSamplingEngine(), ZeroFlow)
    with pytest.raises(RuntimeError, match="needs another reference"):
        pipeline.run()
    assert (tmp_path / "temporal_report.json").is_file()


def test_bad_modulation_shape_is_rejected(tmp_path):
    frame = np.zeros((16, 16, 3), np.uint8)
    mod = np.zeros((16, 16, 2), np.uint8)
    pipeline = CpuTestPipeline(config(tmp_path), Data([frame] * 2, [frame], modulations=[mod] * 2),
                               NnfSamplingEngine(), ZeroFlow)
    with pytest.raises(ValueError, match="modulation"):
        pipeline.run()


def test_router_preserves_disabled_legacy_mode(monkeypatch, tmp_path):
    import importlib
    module = types.ModuleType("ezsynth.legacy_pipeline")
    class Legacy:
        def __init__(self, cfg, data):
            self.config, self.data, self.synthesis_engine = cfg, data, None
        def run(self):
            return "legacy"
    module.SynthesisPipeline = Legacy
    monkeypatch.setitem(sys.modules, "ezsynth.legacy_pipeline", module)
    sys.modules.pop("ezsynth.pipeline", None)
    router = importlib.import_module("ezsynth.pipeline")
    cfg = config(tmp_path)
    cfg.temporal.enabled = False
    assert router.SynthesisPipeline(cfg, None).run() == "legacy"
    sys.modules.pop("ezsynth.pipeline", None)


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("EZSYNTH_TEST_GPU") != "1",
                    reason="requires explicit CUDA build smoke-test opt-in")
def test_real_cuda_synthesis_contract(tmp_path):
    from ezsynth.engines.synthesis_engine import EbsynthEngine
    cfg = config(tmp_path)
    cfg.ebsynth_params.patch_size = 3
    cfg.ebsynth_params.search_vote_iters = 2
    cfg.ebsynth_params.patch_match_iters = 2
    frame = np.random.default_rng(2).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    engine = EbsynthEngine(cfg.ebsynth_params, cfg.pipeline)
    pipeline = CpuTestPipeline(cfg, Data([frame] * 2, [frame]), engine, ZeroFlow)
    output = pipeline.run()
    assert len(output) == 2 and output[1].shape == frame.shape
    np.testing.assert_array_equal(output[0], frame)
