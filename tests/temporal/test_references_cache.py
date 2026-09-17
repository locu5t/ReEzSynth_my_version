from types import SimpleNamespace
import json

import numpy as np
import pytest

from ezsynth.temporal.cache import atomic_json, frame_hash, load_flow, save_flow
from ezsynth.temporal.config import TemporalConfig
from ezsynth.temporal.flow import PairFlow
from ezsynth.temporal.references import choose_references, fuse_candidates, scene_ids


def test_cache_roundtrip(tmp_path):
    flow = np.zeros((4, 5, 2), np.float32)
    path = tmp_path / "pair.npz"
    save_flow(path, flow, "a", "b", "model1")
    np.testing.assert_array_equal(load_flow(path, flow.shape, "a", "b", "model1"), flow)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("src,dst,producer", [("wrong", "b", "p"), ("a", "wrong", "p"), ("a", "b", "wrong")])
def test_cache_rejects_stale_content_and_checkpoint(tmp_path, src, dst, producer):
    path = tmp_path / "pair.npz"
    shape = (4, 5, 2)
    save_flow(path, np.zeros(shape, np.float32), "a", "b", "p")
    with pytest.raises(ValueError):
        load_flow(path, shape, src, dst, producer)


def test_cache_rejects_invalid_shape_and_nonfinite(tmp_path):
    path = tmp_path / "pair.npz"
    save_flow(path, np.zeros((4, 5, 2)), "a", "b", "p")
    with pytest.raises(ValueError):
        load_flow(path, (5, 4, 2), "a", "b")
    with pytest.raises(ValueError):
        save_flow(path, np.full((4, 5, 2), np.nan), "a", "b", "p")


def test_hash_depends_on_pixels_shape_and_dtype():
    image = np.zeros((3, 4, 3), np.uint8)
    assert frame_hash(image) != frame_hash(image + 1)
    assert frame_hash(image) != frame_hash(image.reshape(4, 3, 3))
    assert frame_hash(image) != frame_hash(image.astype(np.float32))


def test_atomic_json_rejects_nan_without_overwriting(tmp_path):
    path = tmp_path / "report.json"
    atomic_json(path, {"old": True})
    with pytest.raises(ValueError):
        atomic_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_shot_selection_never_crosses_cut():
    frames = [np.zeros((8, 8, 3), np.uint8)] * 6
    shots = scene_ids(frames, cuts=[3])
    features = [np.zeros(4)] * 6
    assert choose_references(4, [0, 3], shots, features, [None] * 6, TemporalConfig()) == [3]
    with pytest.raises(ValueError, match="no painted"):
        choose_references(4, [0], shots, features, [None] * 6, TemporalConfig())


def test_optional_cut_detector_and_cut_validation():
    black = np.zeros((32, 32, 3), np.uint8)
    white = np.full_like(black, 255)
    assert scene_ids([black, black, white], detect=True).tolist() == [0, 0, 1]
    with pytest.raises(ValueError):
        scene_ids([black, white], cuts=[2])


def test_face_state_can_select_distant_reference():
    def face(openness):
        return {"face_id": "a", "left": {"openness": openness}, "right": {"openness": openness}}
    faces = [face(1), face(0), face(0)]
    selected = choose_references(1, [0, 2], [0, 0, 0], [np.zeros(3)] * 3,
                                 faces, TemporalConfig(max_references=1))
    assert selected == [2]


def test_max_distance_is_enforced():
    with pytest.raises(ValueError):
        choose_references(5, [0], [0] * 6, [np.zeros(3)] * 6, [None] * 6,
                          TemporalConfig(max_reference_distance=2))


def test_blender_does_not_average_incompatible_blink_images():
    a = np.zeros((2, 2, 3), np.uint8)
    b = np.full_like(a, 255)
    error = np.ones((2, 2), np.float32)
    out, conf = fuse_candidates([a, b], [error, error], [error, error * 0.9])
    np.testing.assert_array_equal(out, a)
    assert conf.min() == 1


def test_blender_softly_combines_agreeing_images():
    a = np.full((2, 2, 3), 100, np.uint8)
    b = np.full_like(a, 110)
    error = np.ones((2, 2), np.float32)
    out, _ = fuse_candidates([a, b], [error, error], [error, error])
    assert (out == 105).all()


def test_zero_confidence_fallback_remains_flagged():
    a = np.full((2, 2, 3), 10, np.uint8)
    b = np.full_like(a, 200)
    error = np.ones((2, 2), np.float32)
    out, conf = fuse_candidates([a, b], [error * 5, error], [error * 0, error * 0])
    np.testing.assert_array_equal(out, b)
    assert not conf.any()


@pytest.mark.parametrize("kwargs", [{"max_references": 0}, {"eye_state_tolerance": 0},
                                    {"scene_cuts": [2, 2]}, {"unknown_option": True}])
def test_config_rejects_invalid_or_misspelled_options(kwargs):
    with pytest.raises(ValueError):
        TemporalConfig(**kwargs)


def test_opencv_flow_and_cache_reuse(tmp_path):
    frame = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    project = SimpleNamespace(cache_dir=str(tmp_path), force_recompute_flow=False)
    provider = PairFlow([frame, frame.copy()], TemporalConfig(flow_backend="opencv"), project, None)
    provider.prepare([(0, 1), (1, 0)])
    assert provider.engine is None  # released before synthesis
    assert provider.read_only
    np.testing.assert_allclose(provider.get(0, 1), 0, atol=1e-3)
    np.testing.assert_allclose(provider.get(1, 0), 0, atol=1e-3)


def test_precomputed_cache_is_never_silently_recomputed(tmp_path):
    frame = np.zeros((16, 16, 3), np.uint8)
    project = SimpleNamespace(cache_dir=str(tmp_path), force_recompute_flow=False)
    provider = PairFlow([frame, frame], TemporalConfig(flow_backend="precomputed",
                        precomputed_flow_dir=str(tmp_path)), project, None)
    with pytest.raises(ValueError, match="missing or invalid"):
        provider.get(0, 1)
