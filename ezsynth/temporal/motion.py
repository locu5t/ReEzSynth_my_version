"""Explicit pull-warp semantics. Flow at target (x,y) points into source pixels."""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def pixel_grid(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    return np.stack((x, y), -1).astype(np.float32)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA warping was requested but CUDA is unavailable")
    if device not in ("cpu", "cuda"):
        raise ValueError("warp device must be auto, cpu, or cuda")
    return device


def pull_warp(image: np.ndarray, target_to_source: np.ndarray,
              device: str = "cpu", nearest: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Sample source image with true target->source flow; invalid samples are zero.

    Returns float32 values in the INPUT value range, plus an HxW validity mask.
    There is no reflected border, flow sign approximation, or image normalization.
    """
    image = np.asarray(image)
    flow = np.asarray(target_to_source)
    if image.ndim not in (2, 3) or min(image.shape[:2]) < 1:
        raise ValueError("image must be a nonempty HW or HWC array")
    if flow.ndim != 3 or flow.shape[-1] != 2 or min(flow.shape[:2]) < 1:
        raise ValueError("flow must be a nonempty HxWx2 array")
    if not np.isfinite(image).all():
        raise ValueError("image contains non-finite values")
    sh, sw = image.shape[:2]
    th, tw = flow.shape[:2]
    coords = pixel_grid(th, tw) + flow.astype(np.float32)
    valid = (np.isfinite(coords).all(-1) & (coords[..., 0] >= 0)
             & (coords[..., 0] <= sw - 1) & (coords[..., 1] >= 0)
             & (coords[..., 1] <= sh - 1))
    coords = np.where(valid[..., None], coords, 0)
    grid = coords.copy()
    # align_corners=False normalization works for singleton dimensions too.
    grid[..., 0] = (2 * grid[..., 0] + 1) / sw - 1
    grid[..., 1] = (2 * grid[..., 1] + 1) / sh - 1
    was_gray = image.ndim == 2
    source = image[..., None] if was_gray else image
    dev = resolve_device(device)
    with torch.inference_mode():
        source_t = torch.from_numpy(np.ascontiguousarray(source, dtype=np.float32))
        source_t = source_t.permute(2, 0, 1)[None].to(dev)
        grid_t = torch.from_numpy(np.ascontiguousarray(grid))[None].to(dev)
        result = F.grid_sample(source_t, grid_t, align_corners=False,
                               mode="nearest" if nearest else "bilinear",
                               padding_mode="zeros")
        result = result[0].permute(1, 2, 0).cpu().numpy()
    result[~valid] = 0
    return (result[..., 0] if was_gray else result), valid


def resize_flow(flow: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a displacement field AND scale its x/y vector units."""
    import cv2
    if flow.ndim != 3 or flow.shape[-1] != 2 or height < 1 or width < 1:
        raise ValueError("invalid flow or output dimensions")
    h, w = flow.shape[:2]
    result = cv2.resize(flow.astype(np.float32), (width, height))
    result[..., 0] *= width / w
    result[..., 1] *= height / h
    return result


def correspondence_confidence(source: np.ndarray, target: np.ndarray,
                              source_to_target: np.ndarray, target_to_source: np.ndarray,
                              alpha: float = 0.01, beta: float = 0.5,
                              photo_scale: float = 0.15, device: str = "cpu") -> np.ndarray:
    """Heuristic (not calibrated probability) on the TARGET pixel lattice."""
    if alpha <= 0 or beta <= 0 or photo_scale <= 0:
        raise ValueError("confidence scales must be positive")
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError("confidence requires equally sized HWC source and target")
    for field in (source_to_target, target_to_source):
        if field.shape != source.shape[:2] + (2,) or not np.isfinite(field).all():
            raise ValueError("flow shape or values are invalid")
    sampled_forward, valid = pull_warp(source_to_target, target_to_source, device)
    warped_source, photo_valid = pull_warp(source, target_to_source, device)
    residual_sq = np.square(target_to_source + sampled_forward).sum(-1)
    scale = alpha * (np.square(target_to_source).sum(-1)
                     + np.square(sampled_forward).sum(-1)) + beta
    cycle = np.exp(-residual_sq / scale)
    photo_error = np.abs(warped_source - target.astype(np.float32)).mean(-1) / 255.0
    confidence = cycle * np.exp(-photo_error / photo_scale)
    confidence *= valid & photo_valid
    return np.clip(confidence, 0, 1).astype(np.float32)


def position_guide_and_nnf(flow: np.ndarray, confidence: np.ndarray,
                           seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direct target->KEYFRAME coordinates, never one-step or modulo-wrapped."""
    h, w = flow.shape[:2]
    coords = pixel_grid(h, w) + flow
    valid = (np.isfinite(coords).all(-1) & (coords[..., 0] >= 0)
             & (coords[..., 0] <= w - 1) & (coords[..., 1] >= 0)
             & (coords[..., 1] <= h - 1) & (confidence > 0))
    safe = np.where(valid[..., None], coords, 0)
    normalized = safe / np.array([max(w - 1, 1), max(h - 1, 1)], np.float32)
    target = np.concatenate((normalized, np.zeros((h, w, 1), np.float32)), -1)
    pristine = pixel_grid(h, w) / np.array([max(w - 1, 1), max(h - 1, 1)], np.float32)
    source = np.concatenate((pristine, np.zeros((h, w, 1), np.float32)), -1)
    nnf = np.rint(safe).astype(np.int32)
    # Uncovered pixels get fresh candidates, not clamped/reflected old texture.
    rng = np.random.default_rng(seed)
    nnf[~valid, 0] = rng.integers(0, w, int((~valid).sum()))
    nnf[~valid, 1] = rng.integers(0, h, int((~valid).sum()))
    return (np.rint(source * 255).astype(np.uint8),
            np.rint(np.clip(target, 0, 1) * 255).astype(np.uint8), nnf)
