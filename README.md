# AssetForge

AssetForge is a deterministic 2D game-asset pipeline for AI-generated or artist-authored sprite frames. Generators produce candidates; project profiles own the canvas, palette, anchor, animation, validation, and engine-export contracts.

The normalization pipeline is provider-independent. PixelLab, ImageGen, and manually drawn outputs enter as PNG directories; local ComfyUI is the only packaged execution adapter. Every source then uses the same `ingest -> validate -> export` flow.

## Why

Generated frames often look acceptable in isolation but fail at runtime because their placement, palette, bounding-box size, alpha, or frame selection drifts. AssetForge makes those properties explicit and testable before assets reach the game project.

Key guarantees:

- Fixed-canvas sprites can preserve every original pixel coordinate.
- Only the requested animation is selected from mixed input directories.
- Border-connected backgrounds are removed without erasing enclosed face or material colors.
- Tiny enclosed transparency defects can be repaired under a profile limit; larger holes fail validation.
- Palette limits, foot-anchor drift, bounding-box size drift, internal alpha holes, canvas size, and frame-count gates are evaluated per clip.
- Web registry JSON and Godot `SpriteFrames` exports include the referenced PNG files.
- Writes into a game project require an explicit `--deploy-dir`.

## Install

```bash
git clone https://github.com/mindsurf0176-ui/assetforge.git
cd assetforge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

AssetForge is not published to PyPI yet; the command above installs it from this source checkout.

Verify the installation:

```bash
assetforge --version
assetforge profiles
python -m unittest discover -s assetforge/tests -v
```

## Quick start

Create a provider-independent generation plan:

```bash
assetforge plan \
  --profile web-pixel-demo \
  --character companion \
  --tier village \
  --animation walk \
  --direction south \
  --reference /path/to/master.png \
  --write work/companion-walk-plan.json
```

Normalize, validate, and export an existing frame directory:

```bash
assetforge build \
  --profile web-pixel-demo \
  --input /path/to/raw-walk-frames \
  --work ../assetforge-output/work/companion-walk \
  --output ../assetforge-output/companion-walk.json \
  --character companion \
  --tier village \
  --animation walk \
  --direction south \
  --resource-prefix ./assets/companion/walk/south
```

By default, export creates a self-contained artifact beside the output file. To deploy directly into a game project, provide both a resource prefix and the matching local directory:

```bash
cd /path/to/game
assetforge export \
  --profile godot-pixel-demo \
  --input /path/to/normalized/guardian-walk \
  --output /path/to/game/guardian_walk.tres \
  --character guardian \
  --tier battle-approved \
  --animation walk \
  --direction east \
  --resource-prefix res://assets/sprites/guardian/walk \
  --deploy-dir /path/to/game/assets/sprites/guardian/walk
```

The packaged demo profiles use the current working directory as `projectRoot`, so direct deployment must run from the target game root. A custom profile can set a different project root.

## Input naming

Animation-specific names are preferred:

```text
walk_00.png
walk_01.png
walk_02.png
walk_03.png
```

Provider-neutral names such as `frame_00.png` and `pose_00.png` are also accepted. If a directory contains only another known clip, such as `attack_*.png`, a request for `walk` fails instead of silently exporting the wrong animation.

## Profiles

Profiles are JSON contracts. Two packaged examples are included:

- `web-pixel-demo`: 40x40 fixed-canvas sprites and web registry export.
- `godot-pixel-demo`: `battle-approved` preserves final frame coordinates, while `battle-generated` normalizes 128px generation candidates before Godot `SpriteFrames` export.

Pass a custom profile by file path:

```bash
assetforge doctor --profile /path/to/my-project-profile.json
```

Set `ASSETFORGE_PROFILE_DIR` to replace the built-in profile directory for `assetforge profiles` and profile IDs. Set `ASSETFORGE_COMFY_URL` to override a profile's local ComfyUI endpoint.

The profile schema is packaged at `assetforge/profiles/assetforge-profile.schema.json`.

## Local ComfyUI

`comfy-compile`, `comfy-submit`, `comfy-run`, and `comfy-build` use API-format workflow templates. Network submission is dry-run by default and requires `--execute`.

Low-resolution RGBA runtime sprites can take the lossless preserve path. High-resolution master art can use the configured diffusion workflow. Multi-frame clips should be supplied as a frame directory so all frames share one validation contract.

## Design boundary

AssetForge does not decide gameplay state, timing, or combat outcomes. Those remain deterministic runtime code. It produces regulated visual assets and engine resources that gameplay can reference safely.

Generated images, model weights, game-specific profiles, and private project assets are intentionally not part of this repository.

## Development

```bash
python -m unittest discover -s assetforge/tests -v
```

The current suite covers animation selection, placement preservation, alpha-hole repair, identity and anchor overrides, isolated exports, explicit deployment, web registry output, and Godot `SpriteFrames` output.

## License

AssetForge is released under the [MIT License](LICENSE).
