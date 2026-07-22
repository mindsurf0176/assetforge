# AssetForge

AssetForge is a local 2D sprite-animation factory. It turns separated character art into real PNG frame sequences and GIF previews, then normalizes, validates, and exports them for web games or Godot.

The recurring animation path runs locally and does not consume generation credits:

```text
parts or part sheet -> deterministic rig -> idle/walk/attack/hit/death
                    -> shared canvas + palette -> validation -> engine export
```

It is designed to replace the repeatable sprite-animation and delivery portion of a PixelLab-style workflow. It does not pretend that a flat image contains invisible limbs or unseen viewpoints: production mode requires separated parts, while one assembled image uses an explicitly marked coarse auto-rig.

## What works

- Local deterministic cutout animation with no hosted API or model dependency.
- `biped-side`, `quadruped-side`, and `winged-quadruped-side` rigs.
- Built-in `idle`, `walk`, `attack`, `hit`, and `death` motion clips.
- PNG frames, GIF previews, a contact sheet, SHA-256 hashes, and JSON manifests.
- One shared motion canvas across every requested clip, preserving lunges, recoil, and falls.
- Character-wide palette locking and fixed-canvas normalization through project profiles.
- Web registry JSON and Godot `SpriteFrames` exports with referenced PNG files.
- Strict `RigSpec` and `AnimationSpec` validation, including safe relative part paths and acyclic skeleton checks.
- Existing AI-generated or hand-drawn frame directories can still use the original `ingest -> validate -> export` pipeline.

## Input quality boundary

| Input | Reported quality | Practical use | Limitation |
| --- | --- | --- | --- |
| Named transparent part PNGs | `production` | Full built-in clip set | Art must be separated once |
| One disconnected part-sheet PNG plus mapping | `production` | Full built-in clip set | Flat border background and separated components required |
| One assembled character PNG | `coarse` | Preview, idle, small motion | No hidden-pixel or occlusion synthesis |
| One image expected to become new viewpoints | unsupported | — | Supply direction-specific art or use a generative model |

The `direction` value binds the compiled rig to its source view; it does not invent a new camera angle. For four- or eight-direction games, prepare and animate one part set per required direction. A profile may explicitly reuse a bound direction through `mirrorDirections`, in which case AssetForge mirrors the complete motion horizontally.

## Install

```bash
git clone https://github.com/mindsurf0176-ui/assetforge.git
cd assetforge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

AssetForge is not published to PyPI yet. Verify a source installation with:

```bash
assetforge --version
assetforge rig-archetypes
python -m unittest discover -s assetforge/tests -v
```

## Generate a production animation

Create a directory of transparent PNG parts. File stems are the slot names.

For `biped-side`, production requires `head.png`, `torso.png`, a complete arm set, and a complete leg set:

```text
parts/hero-east/
  head.png
  torso.png
  uarm_f.png  farm_f.png
  uarm_b.png  farm_b.png
  thigh_f.png shin_f.png
  thigh_b.png shin_b.png
```

Single-piece alternatives `arm_f`, `arm_b`, `leg_f`, and `leg_b` are supported. Optional slots include `cape`, `hair_b`, and `weapon`. A combined `legs.png` cannot express the built-in walk and death poses, so it does not satisfy the production gate. If only a complete front-side arm or leg is supplied, AssetForge creates a darker mirrored back-side limb unless `--no-mirror` is set. An incomplete rig fails instead of being mislabeled as production quality.

Generate the five standard clips:

```bash
assetforge animate \
  --parts parts/hero-east \
  --archetype biped-side \
  --character hero \
  --direction east \
  --clips idle,walk,attack,hit,death \
  --height 192 \
  --resample nearest \
  --work build/hero-east
```

For a quadruped, production requires `body.png`, `head.png`, front and hind leg sets; `tail` is optional. `winged-quadruped-side` additionally requires a wing set, while `quadruped-side` never invents one:

```bash
assetforge animate \
  --parts parts/griffin-east \
  --archetype winged-quadruped-side \
  --character griffin \
  --direction east \
  --height 192 \
  --resample nearest \
  --work build/griffin-east
```

Use `nearest` for authored pixel art and `bicubic` for high-resolution painted parts. Frame counts can be overridden within a profile's permitted range:

```bash
assetforge animate ... --frames idle=4,walk=8,attack=6,hit=4,death=8
```

## Use one disconnected part sheet

Use a transparent or flat border-connected background and leave at least 5 pixels between parts; extraction dilates masks to keep anti-aliased edges together. First extract and inspect the connected components:

```bash
assetforge rig-extract \
  --sheet art/griffin-parts.png \
  --output build/griffin-extraction
```

Open `build/griffin-extraction/contact-sheet.png`, then map blob IDs to rig slots:

```json
{
  "blob_01": "body",
  "blob_02": "head",
  "blob_03": "tail",
  "blob_04": "wing_f",
  "blob_05": "foreleg_f",
  "blob_06": "hindleg_f"
}
```

Generate directly from the sheet:

```bash
assetforge animate \
  --part-sheet art/griffin-parts.png \
  --mapping art/griffin-parts-map.json \
  --archetype winged-quadruped-side \
  --character griffin \
  --direction east \
  --height 192 \
  --resample nearest \
  --work build/griffin-east
```

Every extracted blob must appear exactly once in the mapping. Use `"IGNORE"` as the value for a component that is not part of the rig; omissions and aliased duplicate blob IDs fail instead of silently discarding or duplicating art.

## Use one assembled image in coarse mode

This assisted path is useful for evaluating an existing character before its art is separated:

```bash
assetforge animate \
  --reference art/creature-east.png \
  --archetype winged-quadruped-side \
  --character creature \
  --direction east \
  --height 192 \
  --resample nearest \
  --work build/creature-coarse
```

The manifest reports `quality: "coarse"`, `occlusionSynthesis: false`, alpha reconstruction IoU, unscored semantic confidence, omitted regions, and warnings. Reconstruction IoU only proves that visible alpha was preserved; it does not prove that the automatic semantic split was correct. Inspect `rig/rig-overlay.png` and the generated contact sheet. Coarse output can be exported into an isolated review artifact, but `--deploy-dir` is blocked until separated production art is approved.

## Normalize, validate, and export in one run

Add a packaged or custom profile to the same `animate` command:

```bash
assetforge animate \
  --parts parts/griffin-east \
  --archetype winged-quadruped-side \
  --character griffin \
  --direction east \
  --height 192 \
  --resample nearest \
  --profile godot-pixel-demo \
  --tier battle-generated \
  --work build/griffin-godot
```

Validation finishes for every requested clip before any export is written. A successful run produces one self-contained `.tres` or web registry per clip, together with its referenced PNGs.

When `--clips` is omitted with a profile, AssetForge selects the intersection of its built-in local clips and that profile's animation contracts. Unsupported profile-only clips are not fabricated, and unsupported local defaults are not requested.

To deploy into a game project, run from that project's root and authorize the exact destination with both flags:

```bash
assetforge animate \
  --parts /path/to/parts/griffin-east \
  --archetype winged-quadruped-side \
  --character griffin \
  --direction east \
  --profile godot-pixel-demo \
  --tier battle-generated \
  --resource-prefix res://assets/sprites \
  --deploy-dir /path/to/game/assets/sprites \
  --work /path/to/build/griffin-godot
```

AssetForge appends `<character>/<direction>/<clip>` to both roots and verifies every copied frame by hash. It never writes to the configured runtime asset destination implicitly; nested build artifacts remain allowed inside a repository.

An explicit multi-clip deployment is staged and verified in full, then the character-direction tree is swapped as one rollback-safe unit. The tree carries `.assetforge-deployment.json`; later successful runs synchronize it to the newly requested clip set, so removed clips do not linger. A non-empty destination without that ownership marker is refused and preserved.

## Reuse a compiled rig

The first production run writes `build/hero-east/rig/rig.json`. Re-render it without rebuilding the art layout:

```bash
assetforge animate \
  --rig build/hero-east/rig/rig.json \
  --character hero \
  --direction east \
  --clips idle,walk,attack,hit,death \
  --resample nearest \
  --work build/hero-east-rerender
```

For repeatable builds, place an `AnimationSpec` beside the rig directory:

```json
{
  "schemaVersion": 1,
  "id": "hero-east",
  "character": "hero",
  "direction": "east",
  "rig": "rig/rig.json",
  "clips": {
    "idle": {"frames": 6},
    "walk": {"frames": 8},
    "attack": {"frames": 6},
    "hit": {"frames": 4},
    "death": {"frames": 8}
  },
  "render": {
    "renderer": "local-cutout-v1",
    "fit": "shared-motion-bounds",
    "resample": "nearest"
  }
}
```

```bash
assetforge animate --spec hero-east.animation.json
```

`--spec` owns all animation settings. Only `--work` may override its default build directory, so conflicting flags fail instead of being silently ignored.

## Output layout

Production part and part-sheet inputs use this layout. Coarse reference input writes `autorig-report.json` in place of `rig-report.json`.

```text
build/hero-east/
  rig/
    rig.json
    bindpose.png
    rig-overlay.png
    rig-report.json
    parts/*.png
  raw/east/
    idle/*.png
    walk/*.png
    attack/*.png
    hit/*.png
    death/*.png
    idle.gif ... death.gif
    contact-sheet.png
    animation-manifest.json
  normalized/east/<clip>/*.png   # when a profile is used
  reports/east/<clip>.json       # when a profile is used
  exports/*                      # when a profile is used
  character-manifest.json
```

Loop clips sample `[0, 1)` so the first pose is not duplicated. Non-loop clips sample `[0, 1]`, preserving the authored final attack, hit, or death pose.

Built-in walk, attack, hit, and death clips declare `grounded: true`, keeping their composed silhouette on the standing ground line while joints rotate. Custom RigSpecs should use the same flag for grounded actions; profile `contentMin` still rejects any motion envelope that would make individual frames unreadably small.

## Existing frame pipeline

AssetForge also regulates frame directories from ImageGen, ComfyUI, another generator, or an artist:

```bash
assetforge build \
  --profile web-pixel-demo \
  --input /path/to/raw-walk-frames \
  --work build/normalized-walk \
  --output build/companion-walk.json \
  --character companion \
  --tier village \
  --animation walk \
  --direction south
```

Animation-specific names such as `walk_00.png` are preferred. Provider-neutral names such as `frame_00.png` are accepted when the directory does not contain a conflicting known clip.

Local ComfyUI remains available through `comfy-compile`, `comfy-submit`, `comfy-run`, and `comfy-build`. Network submission is dry-run by default and requires `--execute`.

## Profiles and contracts

The packaged profiles are:

- `web-pixel-demo`: 40x40 fixed-canvas sprites and web registry export.
- `godot-pixel-demo`: generated or approved battle sprites and Godot `SpriteFrames` export.

Pass a custom profile by file path. Set `ASSETFORGE_PROFILE_DIR` to replace the built-in profile directory and `ASSETFORGE_COMFY_URL` to override a profile's local ComfyUI endpoint.

Packaged JSON Schemas:

- `assetforge/profiles/assetforge-profile.schema.json`
- `assetforge/schemas/rig-spec.schema.json`
- `assetforge/schemas/animation-spec.schema.json`

Profiles own canvas, minimum readable content size, palette, composed-frame alpha, source-part alpha, anchor, animation, validation, and export rules. `quality.partAlpha` is separate from final-frame transparency so a project can tune intentional holes in rings, wings, or tails without disabling composed-frame reporting. `tier.contentMin` prevents an oversized motion envelope from silently shrinking frames into unreadable pixels. Gameplay state, hit timing, damage, and combat outcomes remain deterministic game code.

When a profile is active, its FPS is used consistently by previews, manifests, and engine exports. RigSpec and profile loop flags must agree; a mismatch fails before rendering. Successful rebuilds clear superseded work frames and transactionally synchronize the owned deployed direction. Generated work directories also carry a hidden ownership marker; AssetForge refuses to reset an arbitrary non-empty directory that lacks it.

Non-loop clips with real motion require at least two samples. Clips such as attack and hit that return to their starting pose require at least three, so endpoint-only sampling cannot silently produce duplicate stills.

## Development

```bash
python -m pip install build
python -m unittest discover -s assetforge/tests -v
python -m build --wheel
```

The test suite covers deterministic local animation, non-loop final poses, coarse-mode diagnostics, part-sheet extraction, path and graph safety, shared-motion normalization, palette locking, validation, and web/Godot export.

## License

AssetForge is released under the [MIT License](LICENSE). PixelLab is a third-party product; AssetForge is independent and is not affiliated with it.
