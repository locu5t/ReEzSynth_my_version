"""Optional model workers. Run in each model's OWN environment, then render later.

No model checkpoint is downloaded or included by this module. Upstream source
checkouts and weights must be supplied explicitly. See docs/temporal_2026.md.
"""
import argparse
import json
import os
from pathlib import Path
from zipfile import BadZipFile
import subprocess
import sys
from unittest.mock import patch

import cv2
import numpy as np

from .cache import atomic_json, file_hash, frame_hash, load_flow, save_flow
from .motion import resolve_device
from .references import scene_ids

UPSTREAM = {
    "waft": "b152ff1cad1af8c185ee7b141997c48ff3334c87",
    "cotracker": "82e02e8029753ad4ef13cf06be7f4fc5facdda4d",
}


def verified_repository(path, name):
    path = Path(path).resolve(strict=True)
    revision = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, timeout=15).stdout.strip()
    if revision != UPSTREAM[name]:
        raise ValueError(f"{name} must be checked out at {UPSTREAM[name]}, found {revision}")
    dirty = subprocess.run(["git", "-C", str(path), "diff", "--name-only", "HEAD"], check=True,
                           capture_output=True, text=True, timeout=15).stdout.strip()
    if dirty:
        raise ValueError("upstream tracked source has local changes; review before adapting the pin")
    return path


def read_project(config_path):
    import yaml
    from ..config import MainConfig
    from ..data import ProjectData
    config = MainConfig(**yaml.safe_load(Path(config_path).read_text(encoding="utf-8")))
    frames = ProjectData(config.project).get_content_frames()
    hashes = [frame_hash(frame) for frame in frames]
    shots = scene_ids(frames, config.temporal.scene_cuts, config.temporal.detect_scene_cuts,
                      config.temporal.scene_cut_threshold)
    return config, frames, hashes, shots


def face_record(result, height, width, connections, face_id):
    """No face or multiple faces -> unknown. Missing blink scores -> unknown eye."""
    if len(result.face_landmarks) != 1 or len(result.face_blendshapes) != 1:
        return None
    points = result.face_landmarks[0]
    scores = {category.category_name: category.score for category in result.face_blendshapes[0]}
    record = {"face_id": face_id, "left": None, "right": None}
    for side in ("left", "right"):
        score = scores.get("eyeBlink" + side.title())
        if score is None or not np.isfinite(score):
            continue
        edges = getattr(connections, "FACE_LANDMARKS_" + side.upper() + "_EYE")
        ids = sorted({v for edge in edges for v in (edge.start, edge.end)})
        if not ids or max(ids) >= len(points):
            continue
        coordinates = np.array([[points[i].x * (width - 1), points[i].y * (height - 1)]
                                for i in ids], np.float32)
        if not np.isfinite(coordinates).all():
            continue
        coordinates[:, 0] = np.clip(coordinates[:, 0], 0, width - 1)
        coordinates[:, 1] = np.clip(coordinates[:, 1], 0, height - 1)
        hull = cv2.convexHull(coordinates).reshape(-1, 2)
        if len(hull) < 3:
            # A closed eye may be nearly collinear: preserve a narrow eyelid box.
            lo, hi = coordinates.min(0), coordinates.max(0)
            lo = np.maximum(lo - [0, 1], 0)
            hi = np.minimum(hi + [0, 1], [width - 1, height - 1])
            hull = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        record[side] = {"openness": float(np.clip(1 - score, 0, 1)), "polygon": hull.tolist()}
    return record


def faces_job(args, config, frames, hashes, shots):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
    model_path = Path(args.model).resolve(strict=True)
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE, num_faces=2, output_face_blendshapes=True)
    results = []
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for index, frame in enumerate(frames):
            rgb = np.ascontiguousarray(frame[..., ::-1])
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            results.append(face_record(result, *frame.shape[:2], FaceLandmarksConnections,
                                       f"single-subject-shot-{int(shots[index])}"))
    atomic_json(args.output, {"schema": 1, "frame_hashes": hashes, "frames": results,
                             "producer": "mediapipe FaceLandmarker; 1-eyeBlink is a heuristic",
                             "model_sha256": file_hash(model_path), "mediapipe_version": mp.__version__})


def cotracker_job(args, config, frames, hashes, shots):
    if not args.acknowledge_noncommercial:
        raise ValueError("Review CoTracker's CC-BY-NC license; explicit --acknowledge-noncommercial is required")
    repository = verified_repository(args.repository, "cotracker")
    checkpoint = Path(args.model).resolve(strict=True)
    sys.path.insert(0, str(repository))
    import torch
    from cotracker.predictor import CoTrackerPredictor
    device = resolve_device(args.device)
    model = CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, v2=False, window_len=60)
    model = model.to(device).eval()
    h, w = frames[0].shape[:2]
    scale = min(1.0, args.max_side / max(h, w))
    rh, rw = max(2, round(h * scale)), max(2, round(w * scale))
    if h < 2 or w < 2:
        raise ValueError("CoTracker requires both image dimensions greater than one")
    parts = []
    keys = config.project.style_indices
    for shot in np.unique(shots):
        indices = np.flatnonzero(shots == shot)
        if len(indices) < 2:
            continue
        if len(indices) > args.max_frames:
            raise ValueError(f"Shot {shot} has {len(indices)} frames, over --max-frames={args.max_frames}; "
                             "use a shorter clip or explicitly raise the bound after checking VRAM")
        query_frames = [int(k - indices[0]) for k in keys if k in indices]
        if not query_frames:
            raise ValueError(f"Shot {shot} needs a painted keyframe")
        gx, gy = np.meshgrid(np.linspace(0, rw - 1, args.grid_size),
                             np.linspace(0, rh - 1, args.grid_size))
        xy = np.stack((gx.ravel(), gy.ravel()), -1)
        queries = np.concatenate([np.column_stack((np.full(len(xy), t), xy)) for t in query_frames])
        clip = np.stack([cv2.resize(frames[i], (rw, rh))[..., ::-1] for i in indices])
        with torch.inference_mode():
            video = torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2)[None].float().to(device)
            query = torch.from_numpy(queries.astype(np.float32))[None].to(device)
            tracks, visibility = model(video, queries=query, backward_tracking=True)
            tracks = tracks[0].float().cpu().numpy()
            visibility = visibility[0].float().cpu().numpy()
        if visibility.ndim == 3 and visibility.shape[-1] == 1:
            visibility = visibility[..., 0]
        if tracks.shape != (len(indices), len(queries), 2) or visibility.shape != tracks.shape[:2]:
            raise ValueError("Unexpected CoTracker output contract; check the pinned source")
        tracks *= np.array([(w - 1) / (rw - 1), (h - 1) / (rh - 1)], np.float32)
        part_tracks = np.zeros((len(frames), len(queries), 2), np.float32)
        part_visibility = np.zeros((len(frames), len(queries)), np.float32)
        part_tracks[indices], part_visibility[indices] = tracks, visibility
        parts.append((part_tracks, part_visibility))
    tracks = np.concatenate([p[0] for p in parts], 1) if parts else np.zeros((len(frames), 0, 2), np.float32)
    visibility = np.concatenate([p[1] for p in parts], 1) if parts else np.zeros((len(frames), 0), np.float32)
    if not np.isfinite(tracks).all() or not np.isfinite(visibility).all():
        raise ValueError("CoTracker produced non-finite results")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Opening the requested path avoids NumPy silently appending another suffix.
    with output.open("wb") as handle:
        np.savez_compressed(handle, schema=1, frame_hashes=hashes, tracks=tracks,
                            visibility=visibility, revision=UPSTREAM["cotracker"],
                            checkpoint_sha256=file_hash(checkpoint))


def waft_job(args, config, frames, hashes, shots):
    repository = verified_repository(args.repository, "waft")
    checkpoint = Path(args.model).resolve(strict=True)
    waft_config = Path(args.waft_config).resolve(strict=True)
    settings = json.loads(waft_config.read_text(encoding="utf-8"))
    if settings.get("algorithm") != "waft-a1":
        raise ValueError("This worker supports the inspected WAFT a1 adaptation only")
    pairs = set()
    for index in range(len(frames)):
        if index in config.project.style_indices:
            continue
        for key in config.project.style_indices:
            if not 0 <= key < len(frames):
                raise ValueError("keyframe index outside video")
            if shots[key] == shots[index] and (config.temporal.max_reference_distance is None
                    or abs(key - index) <= config.temporal.max_reference_distance):
                pairs.update(((key, index), (index, key)))
    if len(pairs) > args.max_pairs:
        raise ValueError(f"{len(pairs)} directional pairs exceeds --max-pairs={args.max_pairs}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(repository))
    import torch
    import timm
    from model import fetch_model
    from model.waft_a1 import DepthAnythingFeature
    from inference_tools import InferenceWrapper
    create_model = timm.create_model

    def without_pretrained_download(*positional, **kwargs):
        kwargs["pretrained"] = False
        return create_model(*positional, **kwargs)

    # The FULL WAFT checkpoint must contain all parameters. Avoid downloading or
    # loading redundant backbone checkpoints, then demand strict coverage below.
    with patch("timm.create_model", without_pretrained_download), patch(
            "model.waft_a1.DepthAnythingFeature",
            lambda encoder: DepthAnythingFeature(encoder=encoder, pretrained=False)):
        model = fetch_model(argparse.Namespace(**settings))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    device = resolve_device(args.device)
    model = model.to(device).eval()
    wrapped = InferenceWrapper(model, scale=0.0, train_size=settings["image_size"],
                               pad_to_train_size=False, tiling=False)
    producer = json.dumps({"name": "WAFT-a1", "revision": UPSTREAM["waft"],
                           "checkpoint_sha256": file_hash(checkpoint),
                           "config_sha256": file_hash(waft_config)}, sort_keys=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for source, target in sorted(pairs):
        path = output / f"{source:05d}_to_{target:05d}.npz"
        shape = frames[source].shape[:2] + (2,)
        try:
            load_flow(path, shape, hashes[source], hashes[target], producer)
            continue
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            pass
        with torch.inference_mode():
            images = [torch.from_numpy(np.ascontiguousarray(frames[i][..., ::-1]))
                      .permute(2, 0, 1)[None].float().to(device) for i in (source, target)]
            field = wrapped.calc_flow(*images)["flow"][-1][0].permute(1, 2, 0).float().cpu().numpy()
        if field.shape != shape:
            raise ValueError("WAFT output resolution does not match source frames")
        save_flow(path, field, hashes[source], hashes[target], producer)
    atomic_json(output / "manifest.json", {"schema": 1, "frame_hashes": hashes,
                                           "producer": producer, "pair_count": len(pairs)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["faces", "cotracker", "waft"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="Explicit local .task/.pth checkpoint; no automatic download")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository", help="Local pinned upstream checkout for WAFT/CoTracker")
    parser.add_argument("--waft-config", help="WAFT a1 JSON configuration")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--acknowledge-noncommercial", action="store_true")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--max-pairs", type=int, default=2000)
    args = parser.parse_args()
    if args.backend != "faces" and not args.repository:
        parser.error("--repository is required for this backend")
    if args.backend == "waft" and not args.waft_config:
        parser.error("--waft-config is required")
    if min(args.grid_size, args.max_side, args.max_frames, args.max_pairs) < 2:
        parser.error("grid/size/frame/pair bounds must be at least two")
    config, frames, hashes, shots = read_project(args.config)
    {"faces": faces_job, "cotracker": cotracker_job, "waft": waft_job}[args.backend](
        args, config, frames, hashes, shots)


if __name__ == "__main__":
    main()
