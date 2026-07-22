# Changelog

## 0.3.0

- Added a fully local deterministic cutout renderer that generates real PNG frames and GIF previews.
- Added `biped-side`, `quadruped-side`, and `winged-quadruped-side` rigs with idle, walk, attack, hit, and death clips.
- Added production input from named transparent parts or a disconnected part sheet with explicit blob mapping.
- Added an assisted single-reference auto-rig marked `coarse`, with rest-pose diagnostics and honest occlusion warnings.
- Added strict, packaged `RigSpec` and `AnimationSpec` JSON Schemas and reusable compiled rigs.
- Added shared motion bounds so lunges, recoil, and final death poses survive normalization.
- Added character-wide palette locking and one-command profile validation plus per-clip web/Godot export.
- Added nearest-neighbor rig construction and rendering for authored pixel art.
- Bound rigs to validated character and direction identifiers, with explicit profile-driven horizontal mirroring.
- Added production completeness and render-overflow gates, plus coarse-deployment protection.
- Hardened output paths against traversal and destructive writes outside the selected work directory.
- Rejected undersampled recovery actions, non-finite JSON numbers, and RigSpec/profile loop mismatches.
- Made clip ordering deterministic and removed stale work or deployed clips on successful rebuilds.
- Unified profiled preview and engine-export FPS under the profile timing contract.
- Added ownership-marked generated directories and rollback-safe whole-direction deployment transactions.

## 0.2.0

- Added animation-specific frame selection and rejection of mismatched known clips.
- Added fixed-canvas `preservePlacement` mode for pixel-exact approved sprites.
- Added bounded enclosed-alpha repair with a hard validation gate for larger holes.
- Added per-animation identity and foot-anchor overrides.
- Added self-contained web and Godot exports with referenced-frame hash verification.
- Added explicit `--deploy-dir` protection for writes into game projects.
