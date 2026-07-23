# Changelog

## 0.4.0 - 2026-07-23

- Preserve a hand-authored RigSpec canvas verbatim when the selected profile tier uses `preservePlacement=true`.
- Reject RigSpec/profile canvas mismatches and report any nontransparent per-frame overflow instead of silently cropping identity-locked sprites.
- Add `redraw-dataset` and a strict dataset schema for building identity-plus-pose input boards paired with complete animation target boards for a local full-frame redraw model.
- Keep whole-character validation holdouts so repeated frames from one identity cannot be reported as cross-character generalization.
- Add a fail-closed Apple-Silicon MFLUX/FLUX.2 edit backend with dry-run plans, explicit execution, inference low-RAM controls, multiple LoRAs, complete-shard checks, and output verification.
- Publish generated PNG and metadata sidecar pairs with backup-and-rollback so a partial sidecar failure preserves the previous approved result.
- Export redraw pairs in MFLUX flat edit-LoRA layout while keeping validation characters outside the training directory.
- Add quantitative held-out redraw gates for identity, completed cells, pose-guide removal, unused cells, and background drift.
- Add strict MFLUX edit-LoRA dataset validation, train-only subset preparation, base-model shard checks, config planning, and dry-run command compilation.
- Build redraw datasets and promoted frame exports in sibling staging trees, then atomically swap or roll back without destroying the last successful output.
- Add complete validation-holdout batch QC and split only passing boards into hashed, portable, native transparent frame sequences.
- Detect the exact adjacent MFLUX 0.18.0 runtime and gate training on 24 GiB physical memory, 20 GiB free disk, and writable data, cache, and checkpoint paths.
- Add explicit gated edit-LoRA execution that validates an exact reusable config, verifies MFLUX before any subprocess, re-audits before training, and refuses implicit downloads or checkpoint reuse.
- Disable MFLUX's recursive training `low_ram` cache path in managed execution so cache cleanup cannot cross a validated boundary.
- Safely extract the manifest-selected FLUX.2 Klein 4B LoRA from MFLUX checkpoint directories or ZIPs while rejecting archive traversal, symlinks, corrupt tensors, and mismatched metadata.
- Export portable-v2 train-only bundles with out-of-band manifest verification, exact training-file hashes, held-out identity metadata, and either a verified local-model inventory or external model lock.
- Support fail-closed Linux/NVIDIA training with an isolated-GPU inventory check, CUDA 13 MLX version verification, a real float GPU kernel, and verified unquantized base weights.
- Derive whole-epoch schedules from a 1,500-update target by default, with 250-update checkpoints and matching previews for diagnostic checkpoint selection.

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
