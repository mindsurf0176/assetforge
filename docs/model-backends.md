# Model backends

AssetForge treats image generation as a replaceable provider. Every provider must
produce one complete animation sheet plus reproducibility metadata; deterministic
sheet splitting, alpha cleanup, palette locking, anchor fitting, validation, and
export stay inside AssetForge.

## Optional installs

```bash
uv pip install -e '.[vision]'       # OpenCV connected-component helpers
uv pip install -e '.[background]'   # rembg fallback cutout
uv pip install -e '.[generation]'   # Diffusers adapter dependencies
uv pip install -e '.[free]'         # all local, no hosted API backend dependencies
```

The `free` extra installs only local/open-source Python packages. It does not
download model weights or send images to a hosted service. Model checkpoints remain
separate because their license, size, and hardware requirements differ.

SAM 2 is intentionally not a package dependency. It requires a compatible PyTorch
installation and downloaded checkpoints. Use it as an interactive mask-repair
backend, not as the default pixel-art alpha path.

## Provider contract

The provider-facing contract is `SheetRequest` → `SheetResult` in
`assetforge/generation_backends.py`. A provider should keep the identity reference,
camera, scale, palette, and cell layout fixed while generating all poses in one
sheet. It must record the backend, model/checkpoint, seed, prompt, reference, pose
guide, frame count, columns, and rows in `generation-manifest.json`.

The current provider options are:

- Codex Imagegen: use `assetforge codex-prompt` to create the canonical whole-sheet
  prompt, generate one sheet in the built-in image tool, then ingest it with
  `source-sheet --auto-anchor`.
- ComfyUI: use the existing API-format workflow compiler in `providers.py`.
- Diffusers: use IP-Adapter for identity conditioning and ControlNet for pose or
  edge conditioning; map the resulting image to `SheetResult`.
- rembg: optional alpha fallback for non-transparent model outputs.
- SAM 2: optional interactive mask fallback for difficult backgrounds.

## One-shot delivery

```bash
assetforge source-sheet \
  --profile art/characters/moa/moa-v23-profile.json \
  --sheet build/moa-walk-sheet.png \
  --output build/moa-walk-sheet \
  --tier battle-candidate \
  --animation walk \
  --direction east \
  --columns 4 \
  --rows 3 \
  --auto-anchor
```

The output is still a candidate until the configured frame-count and identity
gates pass. A valid canvas is not proof that the generated character stayed
consistent.

For Codex Imagegen, prefer a compact grid such as 4 columns × 3 rows for a
12-frame clip. A 1 × 12 strip is usually too wide for image-generation aspect
ratio limits and is more likely to crop or collapse cells.
For a 14-frame attack, a 4 × 4 grid plus `--frame-count 14` safely ignores the
two trailing empty cells.
