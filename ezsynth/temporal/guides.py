"""Validated, frame-fingerprinted face, tracking, semantic and object guides."""
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .motion import pull_warp


def eye_mask(eye, shape):
    mask = np.zeros(shape, np.uint8)
    if eye is None:
        return mask
    points = np.rint(eye["polygon"]).astype(np.int32)
    cv2.fillPoly(mask, [points], 255)
    # Include the eyelid even when the landmark polygon collapses during a blink.
    radius = max(2, int(np.ptp(points[:, 0]) * 0.12))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    return cv2.dilate(mask, kernel)


def face_guide(face, shape):
    guide = np.zeros(shape + (3,), np.uint8)
    if face is not None:
        for channel, name in enumerate(("left", "right")):
            eye = face.get(name)
            if eye is not None:
                mask = eye_mask(eye, shape)
                guide[..., channel] = mask
                guide[..., 2] = np.maximum(guide[..., 2],
                                          np.rint(mask * eye["openness"]).astype(np.uint8))
    return guide


def eye_compatible(source_face, target_face, eye, tolerance):
    return (source_face is not None and target_face is not None
            and source_face["face_id"] == target_face["face_id"]
            and source_face.get(eye) is not None and target_face.get(eye) is not None
            and abs(source_face[eye]["openness"] - target_face[eye]["openness"]) <= tolerance)


class GuideData:
    def __init__(self, config, frames, hashes):
        self.config = config
        self.hashes = hashes
        self.shape = frames[0].shape[:2]
        self.faces = [None] * len(frames)
        self.tracks = self.visibility = None
        if config.face_annotations:
            data = json.loads(Path(config.face_annotations).read_text(encoding="utf-8"))
            self._manifest(data)
            self.faces = data["frames"]
            if len(self.faces) != len(frames):
                raise ValueError("face annotations must contain one entry per frame")
            for face in self.faces:
                if face is None:
                    continue
                if not isinstance(face.get("face_id"), str) or not face["face_id"]:
                    raise ValueError("each face requires a nonempty face_id")
                for name in ("left", "right"):
                    eye = face.get(name)
                    if eye is None:
                        continue
                    openness = float(eye["openness"])
                    points = np.asarray(eye["polygon"], dtype=np.float32)
                    h, w = self.shape
                    if (not np.isfinite(openness) or not 0 <= openness <= 1
                            or points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                            or not np.isfinite(points).all() or (points < 0).any()
                            or (points[:, 0] > w - 1).any() or (points[:, 1] > h - 1).any()):
                        raise ValueError("invalid openness or pixel-coordinate eye polygon")
                    eye["openness"] = openness
        if config.tracks_path:
            with np.load(config.tracks_path, allow_pickle=False) as data:
                self._manifest({"schema": int(data["schema"].item()),
                                "frame_hashes": data["frame_hashes"].tolist()})
                self.tracks = data["tracks"].astype(np.float32)
                self.visibility = data["visibility"].astype(np.float32)
            if (self.tracks.ndim != 3 or self.tracks.shape[0] != len(frames)
                    or self.tracks.shape[2] != 2 or self.visibility.shape != self.tracks.shape[:2]
                    or not np.isfinite(self.tracks).all() or not np.isfinite(self.visibility).all()
                    or (self.visibility < 0).any() or (self.visibility > 1).any()):
                raise ValueError("tracks must be TxNx2; visibility must be TxN in [0,1]")
        for directory in (config.semantic_guide_dir, config.object_label_dir):
            if directory:
                manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
                self._manifest(manifest)
                for i in range(len(frames)):
                    if not (Path(directory) / f"{i:05d}.npy").is_file():
                        raise FileNotFoundError(f"Missing guide frame {i} in {directory}")

    def _manifest(self, data):
        if data.get("schema") != 1 or data.get("frame_hashes") != self.hashes:
            raise ValueError("guide bundle schema or frame hashes do not match this video")

    @lru_cache(maxsize=8)
    def semantic(self, index):
        if not self.config.semantic_guide_dir:
            return None
        guide = np.load(Path(self.config.semantic_guide_dir) / f"{index:05d}.npy", allow_pickle=False)
        if (guide.ndim != 3 or guide.shape[:2] != self.shape or guide.dtype != np.uint8
                or not 1 <= guide.shape[2] <= 16):
            raise ValueError("semantic guides must be HxWxC uint8 with 1..16 shared channels")
        return np.ascontiguousarray(guide)

    @lru_cache(maxsize=8)
    def labels(self, index):
        if not self.config.object_label_dir:
            return None
        labels = np.load(Path(self.config.object_label_dir) / f"{index:05d}.npy", allow_pickle=False)
        if (labels.shape != self.shape or not np.issubdtype(labels.dtype, np.integer)
                or (labels < -1).any() or (labels > 2 ** 20).any()):
            raise ValueError("object labels must be HW integers: -1 unknown, 0 background, positive IDs")
        return labels.astype(np.int32)

    @staticmethod
    def label_image(labels):
        # Multiplication by an odd value is injective modulo 2**24.
        value = ((labels.astype(np.int64) + 1) * 2654435761) & 0xFFFFFF
        colors = np.stack([(value >> shift) & 255 for shift in (0, 8, 16)], -1).astype(np.uint8)
        colors[labels < 0] = 0
        return colors

    def track_guides(self, source, target):
        if self.tracks is None:
            return None
        h, w = self.shape
        visible = (self.visibility[source] > 0.5) & (self.visibility[target] > 0.5)
        for index in (source, target):
            points = self.tracks[index]
            visible &= ((points[:, 0] >= 0) & (points[:, 0] < w)
                        & (points[:, 1] >= 0) & (points[:, 1] < h))
        output = []
        for index in (source, target):
            guide = np.zeros((h, w, 3), np.uint8)
            for point_id in np.flatnonzero(visible):
                point = tuple(np.rint(self.tracks[index, point_id]).astype(int))
                value = ((int(point_id) + 1) * 2654435761) & 0xFFFFFF
                color = tuple((value >> shift) & 255 for shift in (0, 8, 16))
                cv2.circle(guide, point, 3, color, -1)
            output.append(guide)
        return tuple(output)

    def gate_confidence(self, confidence, source, target, reverse_flow, device):
        confidence = confidence.copy()
        target_face, source_face = self.faces[target], self.faces[source]
        if target_face is not None:
            for eye in ("left", "right"):
                if target_face.get(eye) is not None and not eye_compatible(
                        source_face, target_face, eye, self.config.eye_state_tolerance):
                    confidence[eye_mask(target_face[eye], self.shape) > 0] = 0
        source_labels, target_labels = self.labels(source), self.labels(target)
        if source_labels is not None:
            warped, valid = pull_warp(source_labels, reverse_flow, device, nearest=True)
            known = (target_labels >= 0) & (warped >= 0)
            confidence[valid & known & (np.rint(warped).astype(np.int32) != target_labels)] = 0
        return confidence

    def unsupported_eyes(self, target, references):
        face = self.faces[target]
        if face is None:
            return []
        return [eye for eye in ("left", "right") if face.get(eye) is not None
                and not any(eye_compatible(self.faces[k], face, eye, self.config.eye_state_tolerance)
                            for k in references)]
