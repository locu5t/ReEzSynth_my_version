import numpy as np
import pytest
import torch

from ezsynth.temporal.motion import (correspondence_confidence, pixel_grid,
                                     position_guide_and_nnf, pull_warp, resize_flow)


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (7, 1), (9, 11)])
@pytest.mark.parametrize("channels", [None, 3])
def test_identity_and_singletons(shape, channels):
    full_shape = shape if channels is None else shape + (channels,)
    image = np.arange(np.prod(full_shape), dtype=np.float32).reshape(full_shape)
    actual, valid = pull_warp(image, np.zeros(shape + (2,), np.float32))
    np.testing.assert_allclose(actual, image, atol=1e-4)
    assert valid.all()


def test_true_backward_flow_and_invalid_border():
    image = np.arange(60, dtype=np.float32).reshape(6, 10)
    backward = np.zeros((6, 10, 2), np.float32)
    backward[..., 0] = -2
    output, valid = pull_warp(image, backward)
    np.testing.assert_allclose(output[:, 2:], image[:, :-2], atol=1e-5)
    assert not valid[:, :2].any()
    assert not output[:, :2].any()  # no reflected phantom content


def test_nonuniform_inverse_is_not_negated_forward():
    image = np.arange(8, dtype=np.float32)[None]
    forward = np.zeros((1, 8, 2), np.float32)
    forward[..., 0] = np.arange(8)  # mapping x -> 2*x
    backward = np.zeros_like(forward)
    backward[..., 0] = -np.arange(8) / 2  # correct inverse lives on TARGET grid
    correct, _ = pull_warp(image, backward)
    wrong, _ = pull_warp(image, -forward)
    np.testing.assert_allclose(correct[0], np.arange(8) / 2, atol=1e-5)
    assert not np.allclose(correct, wrong)


def test_invalid_and_nonfinite_flow_samples_are_masked():
    flow = np.zeros((4, 4, 2), np.float32)
    flow[0, 0, 0] = np.nan
    flow[1, 1, 1] = np.inf
    output, valid = pull_warp(np.ones((4, 4), np.float32), flow)
    assert not valid[0, 0] and not valid[1, 1]
    assert np.isfinite(output).all()
    assert output[0, 0] == output[1, 1] == 0


def test_resize_flow_scales_vectors():
    flow = np.ones((4, 8, 2), np.float32)
    resized = resize_flow(flow, 8, 24)
    np.testing.assert_allclose(resized[..., 0], 3)
    np.testing.assert_allclose(resized[..., 1], 2)


def test_position_guide_has_no_modulo_seam():
    flow = np.zeros((5, 6, 2), np.float32)
    source, target, nnf = position_guide_and_nnf(flow, np.ones((5, 6)))
    np.testing.assert_array_equal(source, target)
    assert target[0, -1, 0] == 255
    assert target[-1, 0, 1] == 255
    np.testing.assert_array_equal(nnf, pixel_grid(5, 6))


def test_invalid_nnf_reseeding_is_bounded_and_repeatable():
    flow = np.full((5, 6, 2), 1000, np.float32)
    conf = np.zeros((5, 6), np.float32)
    a = position_guide_and_nnf(flow, conf, seed=17)[2]
    b = position_guide_and_nnf(flow, conf, seed=17)[2]
    np.testing.assert_array_equal(a, b)
    assert (a >= 0).all() and (a[..., 0] < 6).all() and (a[..., 1] < 5).all()
    assert len(np.unique(a.reshape(-1, 2), axis=0)) > 1


def test_cycle_consistency_and_disocclusion():
    frame = np.full((8, 12, 3), 70, np.uint8)
    fwd = np.zeros((8, 12, 2), np.float32)
    fwd[..., 0] = 2
    bwd = -fwd
    good = correspondence_confidence(frame, frame, fwd, bwd)
    np.testing.assert_allclose(good[:, 2:], 1, atol=1e-6)
    assert not good[:, :2].any()
    bad = correspondence_confidence(frame, frame, fwd, np.zeros_like(bwd))
    assert bad.max() < 0.01


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_confidence_rejects_invalid_backend_output(bad):
    frame = np.zeros((3, 3, 3), np.uint8)
    flow = np.zeros((3, 3, 2), np.float32)
    flow[0, 0, 0] = bad
    with pytest.raises(ValueError):
        correspondence_confidence(frame, frame, flow, flow)


def test_nearest_sampling_preserves_label_values():
    labels = np.array([[0, 10, 200, 999999]], np.int32)
    flow = np.zeros((1, 4, 2), np.float32)
    flow[..., 0] = 0.3
    output, valid = pull_warp(labels, flow, nearest=True)
    assert set(output[valid].tolist()).issubset(set(labels.ravel().tolist()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_warp_matches_cpu():
    rng = np.random.default_rng(5)
    image = rng.random((20, 30, 3), dtype=np.float32)
    flow = rng.random((20, 30, 2), dtype=np.float32) - 0.5
    cpu, cpu_valid = pull_warp(image, flow, "cpu")
    gpu, gpu_valid = pull_warp(image, flow, "cuda")
    np.testing.assert_allclose(cpu, gpu, atol=1e-5)
    np.testing.assert_array_equal(cpu_valid, gpu_valid)
