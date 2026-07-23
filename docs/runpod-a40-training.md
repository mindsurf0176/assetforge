# RunPod A40 edit-LoRA runbook

This is the paid-host path for one AssetForge diagnostic run. A 32-pair run is not a production-quality claim. Create one A40 48 GB Pod with a four-hour hard termination limit, and check the live GPU and storage prices before starting.

## 1. Reject an unsuitable Pod before uploading

Run this as soon as the Pod starts:

```bash
set -euo pipefail
nvidia-smi \
  --query-gpu=index,uuid,name,driver_version,memory.total,memory.free,compute_cap \
  --format=csv,noheader,nounits
free -b
df -BG /workspace
```

Continue only when all of these are true:

- exactly one NVIDIA GPU is visible;
- driver major version is at least 580 and compute capability is at least 7.5;
- total and currently free VRAM are both at least 23 GiB;
- physical RAM is at least 24 GiB;
- `/workspace` has at least 40 GiB free before the 15.98 GB model download.

AssetForge repeats the host checks, requires at least 20 GiB to remain free after the model download, and runs a real float kernel through MLX before any MFLUX process. Managed training accepts only unquantized base weights: the supported CUDA MLX package cannot safely attest the quantized backward path. MFLUX training `low_ram` remains disabled because that mode recursively removes its cache directory.

## 2. Pin and transfer from the local Mac

Set the SSH connection values shown by RunPod. Keep the bundle hash printed by `mflux-train-bundle` out of band:

```bash
set -euo pipefail
export RUNPOD_HOST=PASTE_RUNPOD_HOST
export RUNPOD_SSH_PORT=PASTE_EXTERNAL_SSH_PORT
export RUNPOD_SSH_KEY="$HOME/.ssh/id_ed25519"
export ASSETFORGE_COMMIT="$(git -C /path/to/assetforge rev-parse HEAD)"
export BUNDLE_SHA256=PASTE_64_CHARACTER_BUNDLE_SHA256

rsync -a --progress \
  -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
  /path/to/assetforge-redraw-portable/ \
  "root@$RUNPOD_HOST:/workspace/assetforge-redraw-portable/"
```

Mac's built-in rsync supports `--progress`; it does not support rsync 3.x's `--info=progress2`. Do not upload either local 4-bit model. The portable bundle excludes validation boards, but its manifest records the holdout identities and a relocation-safe SHA-256 inventory of every required unquantized model file.

## 3. Install the pinned code and build a host plan

Connect with the same port and key:

```bash
ssh -p "$RUNPOD_SSH_PORT" -i "$RUNPOD_SSH_KEY" "root@$RUNPOD_HOST"
```

On the Pod, paste the commit and bundle hash recorded locally:

```bash
set -euo pipefail
export ASSETFORGE_COMMIT=PASTE_COMMIT
export BUNDLE_SHA256=PASTE_64_CHARACTER_BUNDLE_SHA256
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/mindsurf0176-ui/assetforge.git /workspace/assetforge
git -C /workspace/assetforge checkout "$ASSETFORGE_COMMIT"

python3 -m pip install --upgrade uv
uv tool install --python 3.13 'mflux==0.18.0'
uv tool install --python 3.13 /workspace/assetforge

assetforge --version
mflux-train --help >/dev/null

MODEL_LOCK=/workspace/assetforge/docs/model-locks/flux2-klein-base-4b-a3b4f484.json
MFLUX_EXE="$(readlink -f "$(command -v mflux-train)")"
MFLUX_PY="$(dirname "$MFLUX_EXE")/python"
"$MFLUX_PY" - "$MODEL_LOCK" <<'PY'
import json
import sys
from huggingface_hub import snapshot_download

lock = json.load(open(sys.argv[1], encoding="utf-8"))
snapshot_download(
    repo_id="black-forest-labs/FLUX.2-klein-base-4B",
    revision="a3b4f4849157f664bdbc776fd7453c2783562f4d",
    local_dir="/workspace/FLUX2-klein-base-4B-unquantized",
    allow_patterns=[entry["path"] for entry in lock["files"]],
)
PY

df -BG /workspace
mkdir -p /workspace/training

assetforge mflux-train-plan \
  --bundle /workspace/assetforge-redraw-portable \
  --expected-bundle-sha256 "$BUNDLE_SHA256" \
  --model-path /workspace/FLUX2-klein-base-4B-unquantized \
  --config-output /workspace/training/assetforge-redraw.json \
  --checkpoint-output /workspace/training/run \
  --write-config | tee /workspace/training/plan.json
```

Stop unless the JSON reports `ready: true`, `modelFingerprintVerified: true`, the intended sample count, the intended schedule, and no blockers. For the current 32-board diagnostic, the default schedule is 47 epochs and 1,504 optimizer updates. The config must report `quantize: null`, `low_ram: false`, and the locked unquantized model path.

## 4. Train in a persistent logged session

The approved config must remain unchanged and `/workspace/training/run` must still be absent:

```bash
export PATH="$HOME/.local/bin:$PATH"

tmux new-session -d -s assetforge-train bash -lc "
  export PATH=\"\$HOME/.local/bin:\$PATH\"
  set -o pipefail
  assetforge mflux-train-plan \
    --bundle /workspace/assetforge-redraw-portable \
    --expected-bundle-sha256 '$BUNDLE_SHA256' \
    --model-path /workspace/FLUX2-klein-base-4B-unquantized \
    --config-output /workspace/training/assetforge-redraw.json \
    --checkpoint-output /workspace/training/run \
    --execute 2>&1 | tee /workspace/training/train.log
"

tmux attach -t assetforge-train
```

AssetForge runs MFLUX's parser dry-run, repeats the host, model, bundle, file, tokenizer, and path audits in the same environment, then launches training.

## 5. Extract adapters and preserve evidence

Do not assume the final checkpoint is best. After successful training, extract every non-zero checkpoint and keep each extraction receipt:

```bash
set -euo pipefail
mkdir -p /workspace/training/adapters

for checkpoint in /workspace/training/run/checkpoints/*_checkpoint.zip; do
  step=$(basename "$checkpoint" _checkpoint.zip)
  if [ "$step" = "0000000" ]; then
    continue
  fi
  assetforge mflux-train-extract \
    --checkpoint "$checkpoint" \
    --output "/workspace/training/adapters/${step}.safetensors" \
    | tee "/workspace/training/adapters/${step}.extract.json"
done

(
  cd /workspace/training/adapters
  sha256sum *.safetensors *.extract.json > SHA256SUMS
)

tar -C /workspace/training -czf /workspace/assetforge-redraw-results.tar.gz \
  assetforge-redraw.json plan.json train.log adapters run/preview run/loss
(
  cd /workspace
  sha256sum assetforge-redraw-results.tar.gz > assetforge-redraw-results.tar.gz.sha256
)
tar -tzf /workspace/assetforge-redraw-results.tar.gz >/dev/null
```

Download and verify before deleting any paid resource:

```bash
scp -P "$RUNPOD_SSH_PORT" -i "$RUNPOD_SSH_KEY" \
  "root@$RUNPOD_HOST:/workspace/assetforge-redraw-results.tar.gz" .
scp -P "$RUNPOD_SSH_PORT" -i "$RUNPOD_SSH_KEY" \
  "root@$RUNPOD_HOST:/workspace/assetforge-redraw-results.tar.gz.sha256" .

shasum -a 256 -c assetforge-redraw-results.tar.gz.sha256
tar -tzf assetforge-redraw-results.tar.gz >/dev/null
```

## 6. Generate and gate the whole local holdout

The portable training bundle intentionally contains no validation images. On the local Mac, use the original dataset and the existing 4-bit model only for inference to render every held-out sample with every downloaded adapter:

```bash
set -euo pipefail
export DATASET_MANIFEST=/path/to/original-redraw-dataset/dataset.json
export LOCAL_MODEL=/path/to/FLUX2-klein-base-4B-mlx-4bit
export RESULTS=/path/to/unpacked/assetforge-redraw-results
mkdir -p "$RESULTS/generated" "$RESULTS/quality"

for adapter in "$RESULTS"/adapters/*.safetensors; do
  step=$(basename "$adapter" .safetensors)
  generated="$RESULTS/generated/$step"
  mkdir -p "$generated"

  python3 - "$DATASET_MANIFEST" "$generated" "$LOCAL_MODEL" "$adapter" <<'PY'
import json
import pathlib
import subprocess
import sys

manifest_path = pathlib.Path(sys.argv[1]).resolve()
generated = pathlib.Path(sys.argv[2]).resolve()
model = pathlib.Path(sys.argv[3]).resolve()
adapter = pathlib.Path(sys.argv[4]).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
root = manifest_path.parent

for entry in manifest["mflux"]["holdout"]["entries"]:
    prompt = (root / entry["prompt"]).read_text(encoding="utf-8").strip()
    subprocess.run(
        [
            "assetforge",
            "mflux-redraw",
            "--input", str(root / entry["input"]),
            "--output", str(generated / f"{entry['sample']}.png"),
            "--prompt", prompt,
            "--model-path", str(model),
            "--lora", str(adapter),
            "--lora-scale", "1.0",
            "--execute",
        ],
        check=True,
    )
PY

  if assetforge redraw-quality-batch \
    --manifest "$DATASET_MANIFEST" \
    --generated-dir "$generated" \
    | tee "$RESULTS/quality/${step}.json"; then
    echo "PASS $step"
  else
    echo "FAIL $step"
  fi
done
```

For the current diagnostic, all 16 Gumiho holdout boards must pass before an adapter can be considered for promotion. Compare passing checkpoints visually for identity, outline weight, palette, pose readability, cell order, and loop continuity. Only then split boards into native frames and feed them to the existing `ingest -> validate -> export` pipeline.

## 7. Stop billing

After the archive checksum, extraction receipts, and local holdout outputs are verified, terminate/delete the Pod and any unneeded network volume. Confirm the billing page afterward; a stopped Pod can still incur storage charges. See [RunPod Pod management](https://docs.runpod.io/pods/manage-pods) and [Pod and storage pricing](https://docs.runpod.io/pods/pricing).
