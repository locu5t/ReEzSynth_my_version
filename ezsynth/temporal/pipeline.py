"""Reference-graph synthesis using the existing EbSynth engine (no diffusion)."""
from pathlib import Path
import warnings

import cv2
import numpy as np

from .cache import atomic_json, frame_hash
from .flow import PairFlow
from .guides import GuideData, eye_mask, face_guide
from .motion import (correspondence_confidence, position_guide_and_nnf,
                     pull_warp, resolve_device)
from .references import choose_references, descriptors, fuse_candidates, scene_ids


class TemporalPipeline:
    def __init__(self, config, data, synthesis_engine, flow_factory=PairFlow):
        self.main = config
        self.config = config.temporal
        self.data = data
        self.engine = synthesis_engine
        self.flow_factory = flow_factory
        self.device = resolve_device(self.config.warp_device)

    @staticmethod
    def _validate(frames, styles, indices):
        if not frames or not styles or len(styles) != len(indices):
            raise ValueError("supply content frames and equally many style images/indices")
        shape = frames[0].shape
        if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) < 1:
            raise ValueError("content must contain nonempty HxWx3 BGR frames")
        if any(f.shape != shape or f.dtype != np.uint8 for f in frames + styles):
            raise ValueError("temporal mode requires equal-resolution HxWx3 uint8 content/styles")
        if (any(type(i) is not int or i < 0 or i >= len(frames) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("style_indices must be unique, in-range, zero-based integers")

    def _edges(self, frames):
        # Keep the configured edge engine, including PAGE/PST. Lazy import lets
        # numerical and routing tests run without building the CUDA extension.
        from ..engines.edge_engine import EdgeEngine
        return EdgeEngine(self.main.precomputation.edge_method).compute(frames)

    @staticmethod
    def _masks(masks, frame_count, shape):
        if masks is None:
            return None
        if len(masks) != frame_count:
            raise ValueError("mask count must equal frame count")
        output = []
        for mask in masks:
            if mask.ndim == 3 and mask.shape[-1] in (1, 3):
                mask = mask[..., 0]
            if mask.shape != shape or mask.dtype != np.uint8:
                raise ValueError("project masks must be HW uint8 (0 outside, 255 inside)")
            output.append(mask.astype(np.float32) / 255.0)
        return output

    def _synthesize(self, index, key, frames, style, edges, guides, flow, user_modulation):
        params = self.main.ebsynth_params
        forward, backward = flow.get(key, index), flow.get(index, key)
        confidence = correspondence_confidence(
            frames[key], frames[index], forward, backward,
            self.config.fb_alpha, self.config.fb_beta, self.config.photo_scale, self.device)
        confidence = guides.gate_confidence(confidence, key, index, backward, self.device)
        stable = np.where(confidence >= self.config.confidence_threshold, confidence, 0)
        src_pos, dst_pos, nnf = position_guide_and_nnf(
            backward, stable, self.config.seed + index * len(frames) + key)
        warped, _ = pull_warp(style, backward, self.device)
        warped = np.rint(warped).clip(0, 255).astype(np.uint8)
        h, w = frames[index].shape[:2]
        one = np.ones((h, w), np.float32)
        entries = [
            (edges[key], edges[index], params.edge_weight, one),
            (frames[key], frames[index], params.image_weight, one),
            (src_pos, dst_pos, params.pos_weight, stable),
            (style, warped, params.warp_weight, stable),
        ]
        if self.config.face_annotations:
            entries.append((face_guide(guides.faces[key], (h, w)),
                            face_guide(guides.faces[index], (h, w)),
                            self.config.face_guide_weight, one))
        anchors = guides.track_guides(key, index)
        if anchors is not None:
            entries.append((*anchors, self.config.track_guide_weight, one))
        semantic = guides.semantic(key)
        if semantic is not None:
            target_semantic = guides.semantic(index)
            if semantic.shape != target_semantic.shape:
                raise ValueError("semantic guide channel counts must be shared across frames")
            entries.append((semantic, target_semantic, self.config.semantic_guide_weight, one))
        labels = guides.labels(key)
        if labels is not None:
            entries.append((guides.label_image(labels), guides.label_image(guides.labels(index)),
                            self.config.object_guide_weight, one))
        modulation = np.concatenate([np.repeat(gate[..., None], source.shape[2], axis=2)
                                     for source, _, _, gate in entries], axis=2)
        if user_modulation is not None:
            value = np.asarray(user_modulation)
            if value.ndim == 2:
                value = value[..., None]
            if (value.dtype != np.uint8 or value.shape[:2] != (h, w)
                    or value.shape[2] not in (1, modulation.shape[2])):
                raise ValueError("modulation must be uint8 HW/HWx1 or match all guide channels")
            modulation *= value.astype(np.float32) / 255.0
        output, error = self.engine.run(
            style, guides=[(np.ascontiguousarray(src), np.ascontiguousarray(dst), weight)
                           for src, dst, weight, _ in entries],
            modulation_map=np.ascontiguousarray(np.rint(modulation * 255).astype(np.uint8)),
            initial_nnf=np.ascontiguousarray(nnf), output_nnf=False)
        if output.shape != frames[index].shape or error.shape != (h, w):
            raise ValueError("synthesis engine returned mismatched frame/error dimensions")
        return output, error, confidence

    def run(self):
        frames = list(self.data.get_content_frames())
        styles = list(self.data.get_style_frames())
        indices = list(self.main.project.style_indices)
        self._validate(frames, styles, indices)
        if self.main.ebsynth_params.cost_function != "ssd":
            raise ValueError("Temporal mode is currently validated for cost_function=ssd only")
        if self.main.ebsynth_params.extra_pass_3x3:
            raise ValueError("Disable extra_pass_3x3 in temporal mode; the legacy optional-pass binding needs a separate fix")
        keys = sorted(indices)
        style_bank = dict(zip(indices, styles))  # sort pairs together, never styles independently
        hashes = [frame_hash(frame) for frame in frames]
        guides = GuideData(self.config, frames, hashes)
        shots = scene_ids(frames, self.config.scene_cuts, self.config.detect_scene_cuts,
                          self.config.scene_cut_threshold)
        features = descriptors(frames)
        references = {i: ([i] if i in style_bank else choose_references(
            i, keys, shots, features, guides.faces, self.config)) for i in range(len(frames))}
        shape = frames[0].shape[:2]
        masks = self._masks(self.data.get_mask_frames(), len(frames), shape)
        modulations = self.data.get_modulation_frames()
        if modulations is not None and len(modulations) != len(frames):
            raise ValueError("modulation frame count must match content")
        pairs = [(a, b) for i, refs in references.items() if i not in style_bank
                 for k in refs for a, b in ((i, k), (k, i))]
        flow = self.flow_factory(frames, self.config, self.main.project, self.main.precomputation) if pairs else None
        if flow is not None:
            flow.prepare(pairs)  # model inference finishes BEFORE CUDA synthesis
        edges = self._edges(frames) if pairs else None
        if edges is not None and (len(edges) != len(frames)
                                  or any(e.shape != frames[0].shape or e.dtype != np.uint8 for e in edges)):
            raise ValueError("edge engine must return aligned HxWx3 uint8 guides")
        report = {"schema": 1, "mode": "direct_reference_graph", "flow_backend": self.config.flow_backend,
                  "warp_device": self.device, "confidence_is_calibrated": False,
                  "frame_hashes": hashes, "scene_ids": shots.tolist(), "frames": [],
                  "recommendations": []}
        output_frames = []
        report_path = Path(self.main.project.output_dir) / "temporal_report.json"
        for i in range(len(frames)):
            refs = references[i]
            active = np.ones(shape, bool) if masks is None else masks[i] > 0
            unsupported = guides.unsupported_eyes(i, refs)
            unsupported = [eye for eye in unsupported if np.any(
                (eye_mask(guides.faces[i][eye], shape) > 0) & active)]
            if i in style_bank:
                output = style_bank[i].copy()
                confidence = np.ones(shape, np.float32)
                unsupported = []
            else:
                candidates = [self._synthesize(i, k, frames, style_bank[k], edges, guides, flow,
                                                None if modulations is None else modulations[i])
                              for k in refs]
                output, confidence = fuse_candidates(
                    [c[0] for c in candidates], [c[1] for c in candidates], [c[2] for c in candidates],
                    self.config.blend_disagreement)
            low = (confidence < self.config.confidence_threshold) & active
            fraction = float(low.sum() / max(int(active.sum()), 1))
            entry = {"frame": i, "references": refs, "painted_keyframe": i in style_bank,
                     "unsupported_eyes": unsupported, "low_support_fraction": fraction}
            report["frames"].append(entry)
            if unsupported or fraction > self.config.recommend_fraction:
                report["recommendations"].append(entry.copy())
                if self.config.missing_reference == "error":
                    atomic_json(report_path, report)
                    raise RuntimeError(f"Frame {i} needs another reference; see {report_path}")
            if masks is not None:
                alpha = masks[i][..., None]
                output = np.rint(output * alpha + frames[i] * (1 - alpha)).clip(0, 255).astype(np.uint8)
            output_frames.append(output)
        atomic_json(report_path, report)
        if report["recommendations"]:
            warnings.warn(f"{len(report['recommendations'])} frames have insufficient reference support; "
                          f"review {report_path}", RuntimeWarning)
        return output_frames
