import json

import numpy as np
import pytest

from ezsynth.temporal.cache import frame_hash
from ezsynth.temporal.config import TemporalConfig
from ezsynth.temporal.guides import GuideData, eye_mask


def make_face(openness, face_id="actor"):
    eye = {"openness": openness, "polygon": [[3, 4], [8, 4], [8, 6], [3, 6]]}
    return {"face_id": face_id, "left": eye, "right": None}


def write_faces(tmp_path, frames, faces):
    path = tmp_path / "faces.json"
    path.write_text(json.dumps({"schema": 1, "frame_hashes": [frame_hash(f) for f in frames],
                                "frames": faces}))
    return str(path)


def test_blink_gate_and_missing_reference_detection(tmp_path):
    frames = [np.zeros((16, 16, 3), np.uint8)] * 2
    faces = [make_face(1), make_face(0)]
    path = write_faces(tmp_path, frames, faces)
    guides = GuideData(TemporalConfig(face_annotations=path), frames, [frame_hash(f) for f in frames])
    confidence = guides.gate_confidence(np.ones((16, 16), np.float32), 0, 1,
                                       np.zeros((16, 16, 2), np.float32), "cpu")
    mask = eye_mask(faces[1]["left"], (16, 16)) > 0
    assert not confidence[mask].any()
    assert confidence[~mask].min() == 1
    assert guides.unsupported_eyes(1, [0]) == ["left"]
    assert guides.unsupported_eyes(1, [1]) == []


def test_missing_or_different_face_is_not_assumed_compatible(tmp_path):
    frames = [np.zeros((16, 16, 3), np.uint8)] * 3
    path = write_faces(tmp_path, frames, [None, make_face(0, "other"), make_face(0)])
    guides = GuideData(TemporalConfig(face_annotations=path), frames, [frame_hash(f) for f in frames])
    assert guides.unsupported_eyes(2, [0, 1]) == ["left"]


@pytest.mark.parametrize("bad", [float("nan"), -0.2, 1.5])
def test_invalid_eye_openness_rejected(tmp_path, bad):
    frames = [np.zeros((16, 16, 3), np.uint8)]
    path = write_faces(tmp_path, frames, [make_face(bad)])
    with pytest.raises(ValueError):
        GuideData(TemporalConfig(face_annotations=path), frames, [frame_hash(f) for f in frames])


def test_stale_face_annotations_rejected(tmp_path):
    frame = np.zeros((16, 16, 3), np.uint8)
    path = write_faces(tmp_path, [frame], [make_face(0)])
    with pytest.raises(ValueError, match="hashes"):
        GuideData(TemporalConfig(face_annotations=path), [frame + 1], [frame_hash(frame + 1)])


def test_invisible_tracks_are_not_rendered(tmp_path):
    frames = [np.zeros((16, 16, 3), np.uint8)] * 2
    hashes = [frame_hash(f) for f in frames]
    path = tmp_path / "tracks.npz"
    tracks = np.array([[[5, 5], [12, 12]], [[6, 5], [13, 12]]], np.float32)
    np.savez(path, schema=1, frame_hashes=hashes, tracks=tracks,
             visibility=np.array([[1, 1], [1, 0]], np.float32))
    guides = GuideData(TemporalConfig(tracks_path=str(path)), frames, hashes)
    a, b = guides.track_guides(0, 1)
    assert a[5, 5].any() and b[5, 6].any()
    assert not a[12, 12].any() and not b[12, 13].any()


def test_track_identity_colors_are_stable(tmp_path):
    frames = [np.zeros((16, 16, 3), np.uint8)] * 2
    hashes = [frame_hash(f) for f in frames]
    path = tmp_path / "tracks.npz"
    np.savez(path, schema=1, frame_hashes=hashes,
             tracks=np.array([[[5, 5]], [[8, 8]]], np.float32),
             visibility=np.ones((2, 1), np.float32))
    guides = GuideData(TemporalConfig(tracks_path=str(path)), frames, hashes)
    a, b = guides.track_guides(0, 1)
    np.testing.assert_array_equal(a[5, 5], b[8, 8])


def test_object_mismatch_masks_confidence_and_palette_has_no_short_period(tmp_path):
    frames = [np.zeros((16, 16, 3), np.uint8)] * 2
    hashes = [frame_hash(f) for f in frames]
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": 1, "frame_hashes": hashes}))
    np.save(tmp_path / "00000.npy", np.ones((16, 16), np.int32))
    np.save(tmp_path / "00001.npy", np.full((16, 16), 252, np.int32))
    guides = GuideData(TemporalConfig(object_label_dir=str(tmp_path)), frames, hashes)
    confidence = guides.gate_confidence(np.ones((16, 16)), 0, 1,
                                       np.zeros((16, 16, 2), np.float32), "cpu")
    assert not confidence.any()
    colors = guides.label_image(np.array([[1, 252, -1]], np.int32))
    assert not np.array_equal(colors[0, 0], colors[0, 1])
    assert not colors[0, 2].any()


def test_float_semantic_features_rejected_instead_of_sent_to_uint8_cuda(tmp_path):
    frames = [np.zeros((16, 16, 3), np.uint8)]
    hashes = [frame_hash(frames[0])]
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": 1, "frame_hashes": hashes}))
    np.save(tmp_path / "00000.npy", np.zeros((16, 16, 8), np.float32))
    guides = GuideData(TemporalConfig(semantic_guide_dir=str(tmp_path)), frames, hashes)
    with pytest.raises(ValueError, match="uint8"):
        guides.semantic(0)
