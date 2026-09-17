"""Shot boundaries, reference ranking and conservative per-pixel fusion."""
import cv2
import numpy as np


def scene_ids(frames, cuts=(), detect=False, threshold=0.65):
    n = len(frames)
    cuts = set(cuts)
    if any(i < 1 or i >= n for i in cuts):
        raise ValueError("scene cuts must be indices in [1, frame_count-1]")
    if detect:
        previous = None
        previous_hist = None
        for i, frame in enumerate(frames):
            small = cv2.resize(frame, (64, 64)).astype(np.float32) / 255.0
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            hist = cv2.normalize(hist, hist).flatten()
            if previous is not None:
                # Conservative heuristic; explicit cut indices remain authoritative.
                difference = float(np.abs(small - previous).mean())
                hist_distance = cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if hist_distance > threshold and difference > threshold * 0.35:
                    cuts.add(i)
            previous, previous_hist = small, hist
    ids = np.zeros(n, np.int32)
    for index in sorted(cuts):
        ids[index:] += 1
    return ids


def descriptors(frames):
    return [cv2.resize(frame, (16, 16)).astype(np.float32).reshape(-1) / 255.0
            for frame in frames]


def choose_references(index, keys, shots, features, faces, config):
    candidates = [k for k in keys if shots[k] == shots[index]]
    if config.max_reference_distance is not None:
        candidates = [k for k in candidates if abs(k - index) <= config.max_reference_distance]
    if not candidates:
        raise ValueError(f"Frame {index}: no painted keyframe in this shot/range; add a reference")
    target_face = faces[index]

    def score(k):
        distance = float(np.square(features[k] - features[index]).mean())
        if target_face is not None:
            candidate_face = faces[k]
            if candidate_face is None or candidate_face["face_id"] != target_face["face_id"]:
                distance += 4.0
            else:
                for eye in ("left", "right"):
                    if target_face.get(eye) is not None:
                        if candidate_face.get(eye) is None:
                            distance += 2.0
                        else:
                            distance += 2 * abs(target_face[eye]["openness"]
                                                - candidate_face[eye]["openness"])
        return distance, abs(k - index), k

    ranked = sorted(candidates, key=score)
    # Reserve candidates for independently blinking eyes where possible.
    preferred = []
    if target_face is not None:
        for eye in ("left", "right"):
            if target_face.get(eye) is None:
                continue
            compatible = [k for k in ranked if faces[k] is not None
                          and faces[k]["face_id"] == target_face["face_id"]
                          and faces[k].get(eye) is not None
                          and abs(faces[k][eye]["openness"] - target_face[eye]["openness"])
                          <= config.eye_state_tolerance]
            if compatible and compatible[0] not in preferred:
                preferred.append(compatible[0])
    ordered = preferred + [k for k in ranked if k not in preferred]
    return ordered[:config.max_references]


def fuse_candidates(images, errors, confidences, disagreement=0.12):
    """Fuse aligned TARGET-index results; do not average disagreeing eye images.

    Unsupported pixels use the lowest-error candidate only as a visual fallback.
    Their returned confidence stays zero so the report never calls them valid.
    """
    if not images or not (len(images) == len(errors) == len(confidences)):
        raise ValueError("candidate lists must be nonempty and equally sized")
    shape = images[0].shape
    if (len(shape) != 3 or any(im.shape != shape for im in images)
            or any(e.shape != shape[:2] for e in errors)
            or any(c.shape != shape[:2] for c in confidences)):
        raise ValueError("candidate image/error/confidence shapes must match")
    image = np.stack(images).astype(np.float32)
    error = np.stack(errors).astype(np.float32)
    confidence = np.stack(confidences).astype(np.float32)
    if (not np.isfinite(image).all() or not np.isfinite(error).all()
            or not np.isfinite(confidence).all()):
        raise ValueError("candidate output contains non-finite values")
    error = np.maximum(error, 0)
    confidence = np.clip(confidence, 0, 1)
    positive = error[error > 0]
    scale = max(float(np.median(positive)) if positive.size else 1.0, 1e-6)
    weights = confidence / (1 + error / scale)
    total = weights.sum(0)
    best = weights.argmax(0)
    unsupported = total <= 1e-8
    best[unsupported] = error.argmin(0)[unsupported]
    yy, xx = np.indices(shape[:2])
    winner = image[best, yy, xx]
    # Blend only candidates that agree with the dominant image. Otherwise the
    # average could literally invent a second pupil or a semi-open blink.
    agreement = np.abs(image - winner[None]).mean(-1) / 255.0 <= disagreement
    weights *= agreement
    den = weights.sum(0)
    blended = (image * weights[..., None]).sum(0) / np.maximum(den[..., None], 1e-8)
    blended[den <= 1e-8] = winner[den <= 1e-8]
    return np.rint(blended).clip(0, 255).astype(np.uint8), confidence.max(0)
