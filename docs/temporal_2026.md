# Temporal reference-graph upgrade (experimental)

This is executable first-stage upgrade code, not a vendor dump or a completed
implementation of every research model discussed. It retains ReEzSynth's
existing CUDA/PyTorch PatchMatch engine and adds an **opt-in direct-reference
pipeline**. Keep this PR in draft until real GPU renders have been reviewed.

## What changes

The legacy pipeline is preserved byte-for-byte as `ezsynth/legacy_pipeline.py`.
The public `SynthesisPipeline` routes to it unless `temporal.enabled: true`.
Existing configurations therefore retain their original synthesis behavior;
legacy bugs are not silently altered for existing projects.

In the new mode, each target is matched directly to selected **original painted
keyframes**, with independent keyframe-to-target and target-to-keyframe flow.
It does not negate a forward flow to approximate an inverse, and it does not
repeatedly warp previous generated output. Direct matching avoids that chain's
accumulated error, but can still fail under large viewpoint changes.

The new mode provides:

- True pull warping with PyTorch `grid_sample`, explicit validity masks, and no
  reflected-border invention. CPU and CUDA devices share one implementation.
- Forward/backward cycle and source-image photometric confidence, with temporal
  position/painted-style guides disabled where correspondence is weak.
- Same-shot reference selection, independent left/right blink compatibility,
  optional visible point anchors, semantic guides, and object labels.
- Candidate images and errors aligned to the **same target frame**. Fusion
  avoids averaging candidates with substantially different appearances, which
  could otherwise create a double pupil or a half-open eye.
- Content/checkpoint-aware, atomic flow caching; exact painted-keyframe output;
  project masks; and a `temporal_report.json` listing weak frames and references.
- Optional preprocessing workers for WAFT a1, CoTracker3 offline, and MediaPipe
  Face Landmarker, using supplied local source/checkpoints in separate processes.

Confidence is a heuristic, **not a calibrated probability or proof of correctness**.
Object-label guides are soft matching costs and confidence gates, not hard CUDA
kernel restrictions. The position/reference algorithm, not a new trained model,
is the core improvement implemented here.

## Start with an existing working ReEzSynth installation

Copy `configs/temporal_2026.yml`, set your paths and zero-based keyframe indices,
then run from the repository root:

```console
python run.py --config configs/temporal_2026.yml
```

`flow_backend: legacy` uses the existing configured RAFT/NeuFlow checkpoint.
`opencv` uses CPU DIS for a small model-free diagnostic; it is not claimed to be
competitive with WAFT on difficult camera motion. `precomputed` reads validated
directional fields exported by the WAFT worker or another compatible producer.
No files or checkpoints are downloaded merely by importing the temporal package.

For the Python API, enable the mode after constructing `Ezsynth` and before run:

```python
from ezsynth.temporal import TemporalConfig
# synth = Ezsynth(...) using your existing paths and RunConfig
synth.main_config.temporal = TemporalConfig(enabled=True, flow_backend="legacy")
frames = synth.run()
```

The opt-in pipeline currently accepts SSD and `extra_pass_3x3: false`. It bypasses
the legacy forward/reverse Poisson blender, chain-NNF propagation, and old sparse
corner controls; their settings do not tune this mode. The legacy optional 3x3
pass has a separate mode-binding issue and is deliberately rejected here.
`pipeline.pyramid_levels` and ordinary EbSynth guide/patch settings still apply.

## Blinks and newly visible surfaces

Face data affects BOTH reference selection and per-pixel temporal confidence.
Open-eye references cannot supply trusted temporal support inside a closed-eye
region. Left and right eyes are evaluated independently; a wink need not force
both eyes into the same state.

This does **not** manufacture artist-approved closed-eye texture from an open-eye
painting. Supply suitable open, closing, closed, reopening, and changed-pose
keyframes when needed. Unsupported regions retain a lowest-error visual fallback,
but their confidence remains zero. Read `temporal_report.json`, add a suitable
painted frame, update `style_path`/`style_indices`, and rerun. Set
`missing_reference: error` to stop at the first unsupported result instead.

Set explicit `scene_cuts` at shot transitions. Every shot must have a painted
reference; the system will not transport a character from another shot. Optional
histogram-based cut detection is only a conservative heuristic, not a substitute
for reviewing the cuts. Set `max_reference_distance` to bound expensive or
unreliable long-range matching. All frames are currently buffered in CPU RAM.

## Optional model workers

Run each worker from this repository root with the Python executable belonging
to that model's **isolated environment**. Do not install all upstream dependency
sets into the rendering environment. A model's environment also needs NumPy,
OpenCV, PyTorch, Pydantic v2 and PyYAML to run these helpers. The existing project's
image-loading imports must be available there. Commands below are templates:
replace the example paths with your actual locations. No administrator access,
OS migration, or automatic system modification is performed.

### MediaPipe: source-video blink observations

Install a compatible MediaPipe Tasks package in its environment and obtain the
Face Landmarker `.task` asset from its official documentation:
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python

```console
python -m ezsynth.temporal.precompute faces --config configs/temporal_2026.yml --model path/to/face_landmarker.task --output projects/my_project/guides/faces.json
```

Then set `temporal.face_annotations` to that JSON. The helper uses
`1 - eyeBlinkLeft/Right` as an openness heuristic, not physical eye aperture.
It keeps raw per-frame states rather than smoothing away brief blinks. No face,
multiple faces, or missing blendshape observations are treated as unknown.

**Single-subject clips only:** the helper assigns a per-shot identifier; it does
not perform identity recognition. The current face schema tracks one subject per
frame. For multiple actors, use separate crops/layers and identity-aware external
annotations rather than treating this helper as a multi-face tracker. Inspect blink detections,
especially small faces, profiles, motion blur, and stylized source footage.
MediaPipe source is Apache-2.0; review the chosen model asset's own terms.

### CoTracker3: visible long-range point guides

Official source: https://github.com/facebookresearch/co-tracker

Use a clean local checkout at:
`82e02e8029753ad4ef13cf06be7f4fc5facdda4d`

Install that checkout's requirements in its own environment and supply its
**offline** checkpoint. CoTracker is CC-BY-NC, not a permissively licensed
commercial dependency. Review the upstream license; the acknowledgement flag
records an explicit choice but does not grant rights or change those terms.

```console
python -m ezsynth.temporal.precompute cotracker --config configs/temporal_2026.yml --repository path/to/co-tracker --model path/to/scaled_offline.pth --output projects/my_project/guides/tracks.npz --acknowledge-noncommercial
```

Set `temporal.tracks_path` to that NPZ. Tracks are queried at painted keyframes,
reset at shot boundaries, and only jointly visible source/target points become
guides. By default a shot is bounded to 120 frames and tracking resolution to
512 pixels on its longest side; increase bounds only after profiling memory.
This helper does not implement CoTracker's online streaming mode.

### WAFT a1: direct bidirectional keyframe flow

Official source: https://github.com/princeton-vl/WAFT

Use a clean local checkout at:
`b152ff1cad1af8c185ee7b141997c48ff3334c87`

The inspected upstream recommends the a1 adaptation for downstream applications.
Use its matching a1 JSON configuration and a **full** checkpoint, with upstream
requirements installed separately. The worker rejects a2 configurations instead
of guessing compatibility. It suppresses redundant pretrained-backbone loading
and requires strict full-checkpoint coverage; an incomplete checkpoint fails
rather than silently running partly random weights.

```console
python -m ezsynth.temporal.precompute waft --config configs/temporal_2026.yml --repository path/to/WAFT --waft-config path/to/matching-a1.json --model path/to/full-waft-a1.pth --output projects/my_project/guides/waft
```

Then set:

```yaml
temporal:
  enabled: true
  flow_backend: precomputed
  precomputed_flow_dir: projects/my_project/guides/waft
```

The worker produces BOTH directions for every eligible original-keyframe/target
pair, not negated fields, with a default 2,000-directional-pair limit. Existing
matching outputs can be reused. Keep `max_reference_distance` consistent between
precomputation and rendering. Source code is BSD-3-Clause; inspect the checkpoint
and backbone licenses separately. Runtime, memory use, checkpoint compatibility,
and native Windows behavior still need real model testing. A pinned revision is
reproducible source selection, not a successful-inference certification.

## External guide formats

All bundles require schema `1` and the exact ordered `frame_hashes` calculated by
`ezsynth.temporal.cache.frame_hash` on the decoded BGR uint8 project frames.
Different resolutions, edits, frame ordering, or source content invalidate them.
NumPy inputs are read with `allow_pickle=False`.

**Faces JSON:** `{"schema":1,"frame_hashes":[...],"frames":[...]}`. Each frame is
`null` or `{"face_id":"actor-A","left":eye_or_null,"right":eye_or_null}`.
An eye is `{"openness":0.0,"polygon":[[x,y],...]}` with at least three full-frame
pixel-coordinate points. Openness ranges from 0 (closed) to 1 (open). Left/right
must be defined consistently across references and source frames.

**Tracks NPZ:** scalar `schema`, Unicode `frame_hashes`, float `tracks` of shape
`[T,N,2]` in full-frame x/y pixels, and float/bool `visibility` of shape `[T,N]`
in [0,1]. Point IDs must remain consistent across the sequence.

**Semantic directory:** `manifest.json` with schema/hashes plus `00000.npy`, etc.
Each array is uint8 `[H,W,C]`, C in 1..16. Learned features must use ONE shared
projection and normalization across all frames. Independent per-frame PCA makes
the channels incomparable. No DINOv3 feature extractor is bundled in this PR.

**Object-label directory:** the same manifest/index naming with integer `[H,W]`
arrays, -1 for unknown, 0 background, positive consistent instance IDs up to
2**20. Labels can come from an external segmentation tool; SAM3 extraction is
not implemented here. Numeric IDs are never interpreted as ordinal distances.

**Directional flow NPZ:** `00000_to_00042.npz` means a displacement defined on
frame 0's lattice, pointing into frame 42. Keys: float32 `[H,W,2]` `flow`, scalar
`schema`, scalar Unicode `source_hash`, `target_hash`, and `producer`. Generate
with `cache.save_flow`. A target pull warp needs the opposite file. The renderer
validates content/shape; choose the precomputed directory explicitly to select
a producer/checkpoint. It never substitutes an invalid external field silently.

## Verification and limits

CPU tests exercise true pull direction, invalid borders, resized flow units,
cycle failure, no modulo seam, blink/wink reference gating, shot isolation,
exact keyframes, invalid bundle/cache rejection, project masking, and same-frame
fusion. Integration tests use model doubles. A stationary-pair OpenCV DIS test
runs actual flow inference; this is not an accuracy benchmark.

A separate CPU test environment can be prepared without changing your renderer:

```console
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-temporal-tests.txt
python -m pytest tests/temporal -q
```

The pinned test set requires Python 3.12+. The development run used Python 3.13,
CPU PyTorch 2.10.0, NumPy 2.3.5, OpenCV 4.13.0, and Pydantic 2.13.4.
GPU-only tests are explicitly skipped without CUDA. To exercise the existing
native synthesis engine after installing/building it, set `EZSYNTH_TEST_GPU=1`
and rerun the tests in the working renderer environment. Synthetic tests are
not a substitute for reviewing real clips.

Before merging, render the same short clips with legacy and new modes: a blink,
a wink, head turn with eye occlusion, fast pan, returning object, and a hard cut.
Compare keyframe fidelity, visible flicker, incorrect eye-state frames, boundary
smear, frame count, render time, and peak CPU/GPU memory. Test the selected
pretrained workers separately, recording exact checkpoints and configurations.

**Not implemented:** LivePortrait-generated face references, TAPTR, automatic
SAM3 segmentation, DINOv3 extraction, VGGT camera/depth reconstruction, a 3D
reference graph, hard object-constrained CUDA search, or integrated FastBlend
postprocessing. They are not exposed as fake working backends. This PR also does
not make the whole pipeline GPU-resident or solve texture that was never painted.
Those remain separate follow-on engineering and validation work.
