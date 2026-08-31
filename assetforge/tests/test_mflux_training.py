from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import assetforge.mflux_training as training_module
from assetforge.cli import parser as assetforge_parser
from assetforge.mflux_training import (
    MFLUX_TRAINING_MODEL,
    MFLUX_TRAINING_VERSION,
    MIN_LOCAL_TRAINING_FREE_DISK_GIB,
    build_mflux_training_plan,
    compile_mflux_train_command,
    create_portable_training_bundle,
    discover_mflux_train_executable,
    mflux_training_doctor,
    mflux_cuda_training_probe,
    prepare_training_data,
    run_mflux_training_plan,
    validate_local_mflux_model,
    validate_portable_training_bundle,
    validate_redraw_training_dataset,
    write_mflux_training_config,
)


def _write_triplet(root: Path, index: int, *, color: tuple[int, int, int], size=(64, 64)) -> dict:
    prefix = f"{index:04d}"
    input_path = root / f"{prefix}_in.png"
    target_path = root / f"{prefix}_out.png"
    prompt_path = root / f"{prefix}_in.txt"
    Image.new("RGB", size, color).save(input_path)
    Image.new("RGB", size, tuple(min(channel + 1, 255) for channel in color)).save(target_path)
    prompt_path.write_text(f"redraw sample {index}\n", encoding="utf-8")
    return {
        "index": index,
        "input": input_path,
        "target": target_path,
        "prompt": prompt_path,
    }


def _pixel_digest(path: Path) -> str:
    with Image.open(path) as opened:
        return hashlib.sha256(opened.convert("RGB").tobytes()).hexdigest()


def _write_safetensors(
    path: Path,
    names: tuple[str, ...] = ("weight",),
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    tensors = {}
    data = bytearray()
    for name in names:
        start = len(data)
        data.extend(b"\0\0\0\0")
        tensors[name] = {"dtype": "F32", "shape": [1], "data_offsets": [start, len(data)]}
    if metadata is not None:
        tensors["__metadata__"] = metadata
    header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(len(header).to_bytes(8, "little") + header + data)


def _dataset(
    root: Path,
    *,
    train_count: int = 2,
    holdout_count: int = 1,
    size: tuple[int, int] = (64, 64),
) -> Path:
    train_root = root / "mflux" / "train"
    holdout_root = root / "mflux" / "holdout"
    train_root.mkdir(parents=True)
    holdout_root.mkdir(parents=True)
    train_entries = []
    holdout_entries = []
    samples = []
    for index in range(1, train_count + 1):
        triplet = _write_triplet(train_root, index, color=(20 + index, 40, 60), size=size)
        sample = f"train-creature-{index}"
        canonical_root = root / "samples" / "train" / sample
        canonical_root.mkdir(parents=True)
        canonical_input = canonical_root / "input.png"
        canonical_target = canonical_root / "target.png"
        shutil.copyfile(triplet["input"], canonical_input)
        shutil.copyfile(triplet["target"], canonical_target)
        train_entries.append(
            {
                "index": index,
                "sample": sample,
                "input": f"mflux/train/{triplet['input'].name}",
                "target": f"mflux/train/{triplet['target'].name}",
                "prompt": f"mflux/train/{triplet['prompt'].name}",
                "promptIndex": (index - 1) % 2,
            }
        )
        samples.append(
            {
                "id": sample,
                "split": "train",
                "character": "train-creature",
                "input": f"samples/train/{sample}/input.png",
                "target": f"samples/train/{sample}/target.png",
                "inputPixelSha256": _pixel_digest(canonical_input),
                "targetPixelSha256": _pixel_digest(canonical_target),
            }
        )
    for index in range(1, holdout_count + 1):
        triplet = _write_triplet(holdout_root, index, color=(80 + index, 100, 120), size=size)
        sample = f"holdout-creature-{index}"
        canonical_root = root / "samples" / "validation" / sample
        canonical_root.mkdir(parents=True)
        canonical_input = canonical_root / "input.png"
        canonical_target = canonical_root / "target.png"
        shutil.copyfile(triplet["input"], canonical_input)
        shutil.copyfile(triplet["target"], canonical_target)
        holdout_entries.append(
            {
                "index": index,
                "sample": sample,
                "input": f"mflux/holdout/{triplet['input'].name}",
                "target": f"mflux/holdout/{triplet['target'].name}",
                "prompt": f"mflux/holdout/{triplet['prompt'].name}",
                "promptIndex": (index - 1) % 2,
            }
        )
        samples.append(
            {
                "id": sample,
                "split": "validation",
                "character": "holdout-creature",
                "input": f"samples/validation/{sample}/input.png",
                "target": f"samples/validation/{sample}/target.png",
                "inputPixelSha256": _pixel_digest(canonical_input),
                "targetPixelSha256": _pixel_digest(canonical_target),
            }
        )
    manifest = {
        "schemaVersion": 1,
        "id": "test-redraw",
        "sampleCount": train_count + holdout_count,
        "splits": {"train": train_count, "validation": holdout_count},
        "samples": samples,
        "mflux": {
            "format": "paired-edit-lora-flat-v1",
            "promptCount": 2,
            "train": {
                "path": "mflux/train",
                "sampleCount": train_count,
                "entries": train_entries,
            },
            "holdout": {
                "path": "mflux/holdout",
                "sampleCount": holdout_count,
                "entries": holdout_entries,
            },
        },
    }
    path = root / "dataset.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _model(
    path: Path,
    *,
    indexed: bool = False,
    missing_shard: bool = False,
    quantization_level: int | None = None,
) -> Path:
    path.mkdir(parents=True)
    (path / "README.md").write_text(
        "base_model: black-forest-labs/FLUX.2-klein-base-4B\n",
        encoding="utf-8",
    )
    metadata = (
        {"mflux_version": MFLUX_TRAINING_VERSION, "quantization_level": str(quantization_level)}
        if quantization_level is not None
        else None
    )
    for component in ("text_encoder", "transformer", "vae"):
        root = path / component
        root.mkdir()
        if indexed and component == "transformer":
            _write_safetensors(root / "0.safetensors", ("a",), metadata=metadata)
            if not missing_shard:
                _write_safetensors(root / "1.safetensors", ("b",), metadata=metadata)
            weight_map = {"a": "0.safetensors", "b": "1.safetensors"}
        else:
            _write_safetensors(root / "0.safetensors", metadata=metadata)
            weight_map = {f"{component}.weight": "0.safetensors"}
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}),
            encoding="utf-8",
        )
    tokenizer = path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text(
        json.dumps({"version": "1.0", "model": {}}) + "\n",
        encoding="utf-8",
    )
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2TokenizerFast"}) + "\n",
        encoding="utf-8",
    )
    return path


def _converted_model(path: Path) -> Path:
    return _model(path)


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _versioned_executable(root: Path, version: str = MFLUX_TRAINING_VERSION) -> Path:
    executable = _executable(root / "venv" / "bin" / "mflux-train")
    metadata = (
        root
        / "venv"
        / "lib"
        / "python3.11"
        / "site-packages"
        / f"mflux-{version}.dist-info"
        / "METADATA"
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        f"Metadata-Version: 2.4\nName: mflux\nVersion: {version}\n",
        encoding="utf-8",
    )
    return executable


class MfluxTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._platform = mock.patch("assetforge.mflux_training.sys.platform", "darwin")
        self._platform.start()

    def tearDown(self) -> None:
        self._platform.stop()

    def test_linux_cuda_probe_requires_driver_vram_backend_and_real_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = _versioned_executable(root)
            _executable(executable.parent / "python")
            version = {
                "detected": MFLUX_TRAINING_VERSION,
                "compatible": True,
            }
            nvidia = mock.Mock(
                returncode=0,
                stdout="0, GPU-a40, NVIDIA A40, 580.65.06, 46068, 45000, 8.6\n",
                stderr="",
            )
            mlx = mock.Mock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "mlxVersion": "0.30.3",
                        "cudaPackageVersion": "0.30.3",
                        "cudaAvailable": True,
                        "gpuAvailable": True,
                        "kernelSum": 480.0,
                        "defaultDevice": "Device(gpu, 0)",
                    }
                )
                + "\n",
                stderr="",
            )
            with (
                mock.patch("assetforge.mflux_training.shutil.which", return_value="/usr/bin/nvidia-smi"),
                mock.patch(
                    "assetforge.mflux_training.subprocess.run",
                    side_effect=[nvidia, mlx],
                ) as run,
            ):
                report = mflux_cuda_training_probe(
                    executable,
                    version_report=version,
                    environ={"PATH": "/usr/bin"},
                    platform_name="linux",
                )

            self.assertTrue(report["required"])
            self.assertTrue(report["ready"])
            self.assertTrue(report["gpus"][0]["eligible"])
            self.assertEqual(report["mlx"]["kernelSum"], 480.0)
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(call.kwargs["shell"] is False for call in run.call_args_list))
            self.assertNotIn("device_count", run.call_args_list[1].args[0][3])
            self.assertIn("mx.is_available(mx.gpu)", run.call_args_list[1].args[0][3])
            self.assertNotIn("quantized", run.call_args_list[1].args[0][3].lower())

    def test_linux_cuda_probe_fails_closed_on_ineligible_gpu(self) -> None:
        cases = (
            (
                "0, GPU-a40, NVIDIA A40, 579.99, 46068, 45000, 8.6\n",
                "isolated NVIDIA GPU",
            ),
            (
                "0, GPU-l4, NVIDIA L4, 580.65, 22000, 21000, 8.9\n",
                "isolated NVIDIA GPU",
            ),
            (
                "0, GPU-v100, NVIDIA V100, 580.65, 32768, 32000, 7.0\n",
                "isolated NVIDIA GPU",
            ),
            (
                "0, GPU-a40, NVIDIA A40, 580.65, 46068, 12000, 8.6\n",
                "isolated NVIDIA GPU",
            ),
        )
        for inventory, expected in cases:
            with self.subTest(inventory=inventory), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                executable = _versioned_executable(root)
                _executable(executable.parent / "python")
                with (
                    mock.patch(
                        "assetforge.mflux_training.shutil.which",
                        return_value="/usr/bin/nvidia-smi",
                    ),
                    mock.patch(
                        "assetforge.mflux_training.subprocess.run",
                        return_value=mock.Mock(returncode=0, stdout=inventory, stderr=""),
                    ) as run,
                ):
                    report = mflux_cuda_training_probe(
                        executable,
                        version_report={"detected": MFLUX_TRAINING_VERSION, "compatible": True},
                        environ={"PATH": "/usr/bin"},
                        platform_name="linux",
                    )

                self.assertFalse(report["ready"])
                self.assertTrue(any(expected in blocker for blocker in report["blockers"]))
                self.assertEqual(run.call_count, 1)

    def test_linux_cuda_probe_rejects_an_ambiguous_multi_gpu_host(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = _versioned_executable(root)
            _executable(executable.parent / "python")
            inventory = (
                "0, GPU-small, NVIDIA T4, 580.65, 16000, 15000, 7.5\n"
                "1, GPU-a40, NVIDIA A40, 580.65, 46068, 45000, 8.6\n"
            )
            with (
                mock.patch(
                    "assetforge.mflux_training.shutil.which",
                    return_value="/usr/bin/nvidia-smi",
                ),
                mock.patch(
                    "assetforge.mflux_training.subprocess.run",
                    return_value=mock.Mock(returncode=0, stdout=inventory, stderr=""),
                ) as run,
            ):
                report = mflux_cuda_training_probe(
                    executable,
                    version_report={"detected": MFLUX_TRAINING_VERSION, "compatible": True},
                    environ={"PATH": "/usr/bin"},
                    platform_name="linux",
                )

            self.assertFalse(report["ready"])
            self.assertTrue(any("exactly one GPU" in item for item in report["blockers"]))
            self.assertEqual(run.call_count, 1)

    def test_linux_cuda_probe_rejects_cpu_or_wrong_cuda_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = _versioned_executable(root)
            _executable(executable.parent / "python")
            results = [
                mock.Mock(
                    returncode=0,
                    stdout="0, GPU-a40, NVIDIA A40, 580.65, 46068, 45000, 8.6\n",
                    stderr="",
                ),
                mock.Mock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "mlxVersion": "0.30.3",
                            "cudaPackageVersion": "0.30.2",
                            "cudaAvailable": False,
                            "gpuAvailable": False,
                            "kernelSum": 480.0,
                            "defaultDevice": "Device(cpu, 0)",
                        }
                    )
                    + "\n",
                    stderr="",
                ),
            ]
            with (
                mock.patch("assetforge.mflux_training.shutil.which", return_value="/usr/bin/nvidia-smi"),
                mock.patch("assetforge.mflux_training.subprocess.run", side_effect=results),
            ):
                report = mflux_cuda_training_probe(
                    executable,
                    version_report={"detected": MFLUX_TRAINING_VERSION, "compatible": True},
                    environ={"PATH": "/usr/bin"},
                    platform_name="linux",
                )

            self.assertFalse(report["ready"])
            self.assertTrue(any("invalid attestation" in blocker for blocker in report["blockers"]))

    def test_accepts_real_mflux_component_sharded_layout_without_top_level_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = validate_local_mflux_model(_converted_model(Path(temp) / "model"))

            self.assertTrue(report["ready"])
            self.assertEqual(report["layout"], "mflux-component-sharded")
            self.assertEqual(report["indexedShardCount"], 3)

    def test_diffusers_layout_requires_local_tokenizer_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = _model(root / "model")
            (model / "model_index.json").write_text("{}\n", encoding="utf-8")
            for component in ("text_encoder", "transformer", "vae"):
                (model / component / "config.json").write_text("{}\n", encoding="utf-8")
                (model / component / "model.safetensors.index.json").unlink()
            shutil.rmtree(model / "tokenizer")

            with self.assertRaisesRegex(ValueError, "required tokenizer file"):
                validate_local_mflux_model(model)

    def test_validates_flat_train_and_holdout_manifest_with_low_count_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = validate_redraw_training_dataset(_dataset(root))

            self.assertTrue(report["ok"])
            self.assertEqual(report["train"]["sampleCount"], 2)
            self.assertEqual(report["holdout"]["sampleCount"], 1)
            self.assertEqual(report["train"]["imageSize"], [64, 64])
            self.assertTrue(any("50" in warning for warning in report["warnings"]))

    def test_rejects_sample_count_and_non_flat_triplet_contract_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root)
            value = _read_manifest(manifest_path)
            value["mflux"]["train"]["sampleCount"] = 3
            _write_manifest(manifest_path, value)
            with self.assertRaisesRegex(ValueError, "length must equal sampleCount"):
                validate_redraw_training_dataset(manifest_path)

            value["mflux"]["train"]["sampleCount"] = 2
            value["mflux"]["train"]["entries"][0]["input"] = "mflux/train/nested/0001_in.png"
            _write_manifest(manifest_path, value)
            with self.assertRaisesRegex(ValueError, "flat child"):
                validate_redraw_training_dataset(manifest_path)

    def test_dataset_rejects_an_unexpected_symlink_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root)
            (root / "mflux" / "train" / "alias.png").symlink_to(
                root / "mflux" / "train" / "0001_in.png"
            )

            with self.assertRaisesRegex(ValueError, "unexpected files"):
                validate_redraw_training_dataset(manifest_path)

    def test_rejects_corrupt_png_and_mismatched_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root)
            (root / "mflux" / "train" / "0001_in.png").write_bytes(b"not png")
            with self.assertRaisesRegex(ValueError, "readable PNG"):
                validate_redraw_training_dataset(manifest_path)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root)
            Image.new("RGB", (80, 64)).save(root / "mflux" / "train" / "0001_out.png")
            with self.assertRaisesRegex(ValueError, "sizes differ"):
                validate_redraw_training_dataset(manifest_path)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root)
            for path in root.rglob("*.png"):
                with Image.open(path) as opened:
                    resized = opened.convert("RGB").resize((65, 64))
                resized.save(path)
            value = _read_manifest(manifest_path)
            for sample in value["samples"]:
                sample["inputPixelSha256"] = _pixel_digest(root / sample["input"])
                sample["targetPixelSha256"] = _pixel_digest(root / sample["target"])
            _write_manifest(manifest_path, value)
            with self.assertRaisesRegex(ValueError, "divisible by 16"):
                validate_redraw_training_dataset(manifest_path)

    def test_rejects_holdout_sample_in_train_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root)
            value = _read_manifest(manifest_path)
            value["mflux"]["holdout"]["entries"][0]["sample"] = value["mflux"]["train"]["entries"][0]["sample"]
            _write_manifest(manifest_path, value)

            with self.assertRaisesRegex(ValueError, "canonical validation sample|holdout samples are mixed"):
                validate_redraw_training_dataset(manifest_path)

    def test_rejects_holdout_pixels_copied_under_a_train_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root)
            shutil.copyfile(
                root / "mflux" / "holdout" / "0001_out.png",
                root / "mflux" / "train" / "0001_out.png",
            )

            with self.assertRaisesRegex(ValueError, "pixels do not match|pixels differ"):
                validate_redraw_training_dataset(manifest_path)

    def test_rejects_identical_canonical_target_pixels_across_train_and_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root)
            value = _read_manifest(manifest_path)
            train_sample = value["samples"][0]
            holdout_sample = next(sample for sample in value["samples"] if sample["split"] == "validation")
            train_target = root / train_sample["target"]
            holdout_target = root / holdout_sample["target"]
            shutil.copyfile(train_target, holdout_target)
            shutil.copyfile(train_target, root / "mflux" / "holdout" / "0001_out.png")
            holdout_sample["targetPixelSha256"] = train_sample["targetPixelSha256"]
            _write_manifest(manifest_path, value)

            with self.assertRaisesRegex(ValueError, "identical canonical target pixels"):
                validate_redraw_training_dataset(manifest_path)

    def test_rejects_duplicate_pixels_disguised_as_independent_train_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root)
            value = _read_manifest(manifest_path)
            first, second = [sample for sample in value["samples"] if sample["split"] == "train"]
            for field, digest_field, exported_name in (
                ("input", "inputPixelSha256", "0002_in.png"),
                ("target", "targetPixelSha256", "0002_out.png"),
            ):
                shutil.copyfile(root / first[field], root / second[field])
                shutil.copyfile(root / first[field], root / "mflux" / "train" / exported_name)
                second[digest_field] = first[digest_field]
            _write_manifest(manifest_path, value)

            with self.assertRaisesRegex(ValueError, "independent .* pixels"):
                validate_redraw_training_dataset(manifest_path)

    def test_rejects_character_identity_shared_across_train_and_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root)
            value = _read_manifest(manifest_path)
            holdout = next(sample for sample in value["samples"] if sample["split"] == "validation")
            holdout["character"] = "train-creature"
            _write_manifest(manifest_path, value)

            with self.assertRaisesRegex(ValueError, "validation characters must be held out"):
                validate_redraw_training_dataset(manifest_path)

    def test_rejects_any_canonical_image_pixels_shared_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root)
            value = _read_manifest(manifest_path)
            train_sample = value["samples"][0]
            holdout_sample = next(sample for sample in value["samples"] if sample["split"] == "validation")
            train_input = root / train_sample["input"]
            holdout_input = root / holdout_sample["input"]
            shutil.copyfile(train_input, holdout_input)
            shutil.copyfile(train_input, root / "mflux" / "holdout" / "0001_in.png")
            holdout_sample["inputPixelSha256"] = train_sample["inputPixelSha256"]
            _write_manifest(manifest_path, value)

            with self.assertRaisesRegex(ValueError, "identical canonical image pixels"):
                validate_redraw_training_dataset(manifest_path)

    def test_prepares_deterministic_train_only_smoke_subset_by_byte_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root, train_count=3, holdout_count=1)
            output = root.parent / f"{root.name}-run"

            prepared = prepare_training_data(manifest_path, output, sample_limit=1)

            data_path = Path(prepared["dataPath"])
            self.assertEqual(prepared["mode"], "subset")
            self.assertEqual(prepared["samples"], ["train-creature-1"])
            self.assertEqual(
                {path.name for path in data_path.iterdir()},
                {
                    "0001_in.png",
                    "0001_out.png",
                    "0001_in.txt",
                    "assetforge-training-subset.json",
                },
            )
            self.assertEqual(
                (data_path / "0001_in.png").read_bytes(),
                (root / "mflux" / "train" / "0001_in.png").read_bytes(),
            )
            self.assertNotEqual(
                (data_path / "0001_in.png").read_bytes(),
                (root / "mflux" / "holdout" / "0001_in.png").read_bytes(),
            )
            with self.assertRaisesRegex(FileExistsError, "overwrite is disabled"):
                prepare_training_data(manifest_path, output, sample_limit=1)

    def test_full_training_data_uses_source_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root)
            output = root.parent / f"{root.name}-unused"

            prepared = prepare_training_data(manifest_path, output, sample_limit=None)

            self.assertEqual(prepared["mode"], "source")
            self.assertEqual(Path(prepared["dataPath"]), (root / "mflux" / "train").resolve())
            self.assertFalse(output.exists())

    def test_exports_and_validates_train_only_portable_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = _dataset(root / "dataset", train_count=3, holdout_count=2)
            output = root / "portable"
            model = _model(root / "model")

            created = create_portable_training_bundle(manifest, output, model_path=model)
            report = validate_portable_training_bundle(output)

            self.assertTrue(created["created"])
            self.assertEqual(created["trainSampleCount"], 3)
            self.assertEqual(created["holdoutSampleCount"], 2)
            self.assertFalse(created["holdoutIncluded"])
            self.assertEqual(report["sourceType"], "portable-bundle")
            self.assertEqual(report["train"]["sampleCount"], 3)
            self.assertFalse(report["holdout"]["included"])
            bundle_value = _read_manifest(output / "assetforge-mflux-bundle.json")
            self.assertEqual(bundle_value["schemaVersion"], 2)
            self.assertEqual(bundle_value["format"], "paired-edit-lora-portable-v2")
            self.assertEqual(
                {path.name for path in (output / "data").iterdir()},
                {
                    "0001_in.png",
                    "0001_out.png",
                    "0001_in.txt",
                    "0002_in.png",
                    "0002_out.png",
                    "0002_in.txt",
                    "0003_in.png",
                    "0003_out.png",
                    "0003_in.txt",
                },
            )
            self.assertEqual(
                (output / "data" / "0001_out.png").read_bytes(),
                (root / "dataset" / "mflux" / "train" / "0001_out.png").read_bytes(),
            )
            self.assertNotEqual(
                (output / "data" / "0001_out.png").read_bytes(),
                (root / "dataset" / "mflux" / "holdout" / "0001_out.png").read_bytes(),
            )
            with self.assertRaisesRegex(FileExistsError, "overwrite is disabled"):
                create_portable_training_bundle(manifest, output, model_path=model)

    def test_exports_with_a_trusted_external_model_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = _dataset(root / "dataset")
            model = _model(root / "model")
            local_bundle = root / "local-bundle"
            create_portable_training_bundle(
                manifest,
                local_bundle,
                model_path=model,
            )
            model_lock = json.loads(
                (local_bundle / "assetforge-mflux-bundle.json").read_text(encoding="utf-8")
            )["model"]
            model_lock.pop("included")
            lock_path = root / "model-lock.json"
            lock_path.write_text(json.dumps(model_lock, indent=2) + "\n", encoding="utf-8")

            locked_bundle = root / "locked-bundle"
            result = create_portable_training_bundle(
                manifest,
                locked_bundle,
                model_lock_path=lock_path,
            )
            report = validate_portable_training_bundle(locked_bundle)

            self.assertEqual(
                result["modelFingerprintSha256"],
                model_lock["fingerprintSha256"],
            )
            self.assertEqual(report["modelLock"]["files"], model_lock["files"])
            with self.assertRaisesRegex(ValueError, "exactly one"):
                create_portable_training_bundle(
                    manifest,
                    root / "invalid-bundle",
                    model_path=model,
                    model_lock_path=lock_path,
                )

    def test_portable_bundle_rejects_tampering_and_unexpected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "portable"
            create_portable_training_bundle(
                _dataset(root / "dataset"), output, model_path=_model(root / "model")
            )
            (output / "data" / "0001_in.txt").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from its SHA-256"):
                validate_portable_training_bundle(output)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "portable"
            create_portable_training_bundle(
                _dataset(root / "dataset"), output, model_path=_model(root / "model")
            )
            (output / "data" / "holdout.png").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "unexpected files"):
                validate_portable_training_bundle(output)

    def test_portable_bundle_rejects_source_byte_changes_during_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = _dataset(root / "dataset")
            output = root / "portable"
            original_copy = training_module._copy_exclusive
            changed = False

            def mutate_prompt_then_copy(source: Path, destination: Path) -> None:
                nonlocal changed
                if source.name == "0001_in.txt" and not changed:
                    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                    changed = True
                original_copy(source, destination)

            with (
                mock.patch(
                    "assetforge.mflux_training._copy_exclusive",
                    side_effect=mutate_prompt_then_copy,
                ),
                self.assertRaisesRegex(RuntimeError, "source training file changed"),
            ):
                create_portable_training_bundle(
                    manifest,
                    output,
                    model_path=_model(root / "model"),
                )

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".portable.assetforge-staging-*")), [])

    def test_training_artifact_paths_must_be_disjoint_from_model_and_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = _dataset(root / "dataset")
            model = _model(root / "model")
            executable = _versioned_executable(root)

            with self.assertRaisesRegex(ValueError, "bundle output.*model directory.*disjoint"):
                create_portable_training_bundle(
                    manifest,
                    model / "portable",
                    model_path=model,
                )
            with self.assertRaisesRegex(ValueError, "config_output.*model directory.*disjoint"):
                build_mflux_training_plan(
                    manifest,
                    model_path=model,
                    config_output=model / "train.json",
                    executable=executable,
                    physical_memory_bytes=32 * 1024**3,
                    minimum_free_disk_gib=1.0,
                )
            with self.assertRaisesRegex(ValueError, "config_output.*checkpoint_output.*disjoint"):
                build_mflux_training_plan(
                    manifest,
                    model_path=model,
                    config_output=root / "run" / "train.json",
                    checkpoint_output=root / "run",
                    executable=executable,
                    physical_memory_bytes=32 * 1024**3,
                    minimum_free_disk_gib=1.0,
                )

    def test_portable_bundle_rejects_triplet_and_manifest_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "portable"
            create_portable_training_bundle(
                _dataset(root / "dataset"), output, model_path=_model(root / "model")
            )
            prompt = output / "data" / "0001_in.txt"
            prompt.unlink()
            prompt.symlink_to(output / "data" / "0002_in.txt")
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                validate_portable_training_bundle(output)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "portable"
            create_portable_training_bundle(
                _dataset(root / "dataset"), output, model_path=_model(root / "model")
            )
            wrapper = root / "wrapper"
            wrapper.mkdir()
            (wrapper / "assetforge-mflux-bundle.json").symlink_to(
                output / "assetforge-mflux-bundle.json"
            )
            with self.assertRaisesRegex(ValueError, "manifest.*symbolic link"):
                validate_portable_training_bundle(wrapper)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            actual = root / "actual"
            actual.mkdir()
            output = actual / "portable"
            create_portable_training_bundle(
                _dataset(root / "dataset"), output, model_path=_model(root / "model")
            )
            alias = root / "alias"
            alias.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "bundle path.*symbolic link"):
                validate_portable_training_bundle(alias / "portable")

    def test_execute_revalidates_bundle_after_triplet_injection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "portable"
            model = _model(root / "model")
            create_portable_training_bundle(
                _dataset(root / "dataset"), output, model_path=model
            )
            plan = build_mflux_training_plan(
                portable_bundle=output,
                expected_bundle_sha256=validate_portable_training_bundle(output)["manifestSha256"],
                model_path=model,
                config_output=root / "run" / "train.json",
                executable=_versioned_executable(root),
                physical_memory_bytes=32 * 1024**3,
                minimum_free_disk_gib=1.0,
            )
            write_mflux_training_config(plan)
            _write_triplet(output / "data", 9999, color=(1, 2, 3))
            disk_usage = mock.Mock(free=64 * 1024**3)

            with (
                mock.patch(
                    "assetforge.mflux_training._physical_memory_bytes",
                    return_value=32 * 1024**3,
                ),
                mock.patch(
                    "assetforge.mflux_training.shutil.disk_usage",
                    return_value=disk_usage,
                ),
                self.assertRaisesRegex(RuntimeError, "dataset changed.*unexpected files"),
            ):
                compile_mflux_train_command(plan, execute=True)

    def test_portable_bundle_export_failure_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "portable"
            manifest = _dataset(root / "dataset")
            model = _model(root / "model")
            with (
                mock.patch(
                    "assetforge.mflux_training._copy_exclusive",
                    side_effect=OSError("simulated copy failure"),
                ),
                self.assertRaisesRegex(OSError, "simulated copy failure"),
            ):
                create_portable_training_bundle(manifest, output, model_path=model)

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".portable.assetforge-staging-*")), [])

    def test_portable_bundle_rejects_a_source_changed_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "portable"
            manifest = _dataset(root / "dataset")
            model = _model(root / "model")
            real_copy = training_module._copy_exclusive
            changed = False

            def mutate_then_copy(source: Path, destination: Path) -> None:
                nonlocal changed
                if not changed and source.name == "0001_out.png" and source.parent.name == "train":
                    Image.new("RGB", (64, 64), (250, 1, 2)).save(source)
                    changed = True
                real_copy(source, destination)

            with (
                mock.patch(
                    "assetforge.mflux_training._copy_exclusive",
                    side_effect=mutate_then_copy,
                ),
                self.assertRaisesRegex(RuntimeError, "changed during portable bundle export"),
            ):
                create_portable_training_bundle(manifest, output, model_path=model)

            self.assertTrue(changed)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".portable.assetforge-staging-*")), [])

    def test_builds_training_plan_from_portable_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "portable"
            model = _model(root / "model")
            create_portable_training_bundle(
                _dataset(root / "dataset", train_count=32),
                output,
                model_path=model,
            )
            plan = build_mflux_training_plan(
                portable_bundle=output,
                expected_bundle_sha256=validate_portable_training_bundle(output)["manifestSha256"],
                model_path=model,
                config_output=root / "run" / "train.json",
                executable=_versioned_executable(root),
                physical_memory_bytes=32 * 1024**3,
                minimum_free_disk_gib=1.0,
            )

            self.assertEqual(plan["dataset"]["sourceType"], "portable-bundle")
            self.assertEqual(
                plan["dataset"]["sourceManifestSha256"],
                validate_redraw_training_dataset(root / "dataset" / "dataset.json")["manifestSha256"],
            )
            self.assertEqual(plan["dataset"]["sourceHoldoutCount"], 1)
            self.assertEqual(Path(plan["config"]["data"]), (output / "data").resolve())
            self.assertFalse(plan["dataset"]["holdoutIncluded"])
            self.assertEqual(plan["schedule"]["plannedUpdates"], 1504)
            self.assertTrue(plan["modelFingerprintVerified"])

    def test_portable_plan_requires_the_out_of_band_bundle_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "portable"
            model = _model(root / "model")
            create_portable_training_bundle(
                _dataset(root / "dataset"), output, model_path=model
            )
            arguments = {
                "portable_bundle": output,
                "model_path": model,
                "config_output": root / "run" / "train.json",
                "executable": _versioned_executable(root),
                "physical_memory_bytes": 32 * 1024**3,
                "minimum_free_disk_gib": 1.0,
            }

            with self.assertRaisesRegex(ValueError, "expected_bundle_sha256"):
                build_mflux_training_plan(**arguments)
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_mflux_training_plan(
                    expected_bundle_sha256="0" * 64,
                    **arguments,
                )

    def test_portable_plan_rejects_a_different_model_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "portable"
            locked_model = _model(root / "locked-model")
            create_portable_training_bundle(
                _dataset(root / "dataset"),
                output,
                model_path=locked_model,
            )
            different_model = _model(root / "different-model")
            (different_model / "README.md").write_text(
                "base_model: black-forest-labs/FLUX.2-klein-base-4B\nchanged\n",
                encoding="utf-8",
            )

            plan = build_mflux_training_plan(
                portable_bundle=output,
                expected_bundle_sha256=validate_portable_training_bundle(output)["manifestSha256"],
                model_path=different_model,
                config_output=root / "run" / "train.json",
                executable=_versioned_executable(root),
                physical_memory_bytes=32 * 1024**3,
                minimum_free_disk_gib=1.0,
            )

            self.assertFalse(plan["ready"])
            self.assertFalse(plan["modelFingerprintVerified"])
            self.assertTrue(any("model fingerprint differs" in item for item in plan["blockers"]))

    def test_execute_revalidates_model_fingerprint_after_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "portable"
            model = _model(root / "model")
            create_portable_training_bundle(
                _dataset(root / "dataset"),
                output,
                model_path=model,
            )
            plan = build_mflux_training_plan(
                portable_bundle=output,
                expected_bundle_sha256=validate_portable_training_bundle(output)["manifestSha256"],
                model_path=model,
                config_output=root / "run" / "train.json",
                executable=_versioned_executable(root),
                physical_memory_bytes=32 * 1024**3,
                minimum_free_disk_gib=1.0,
            )
            write_mflux_training_config(plan)
            (model / "README.md").write_text(
                "base_model: black-forest-labs/FLUX.2-klein-base-4B\nchanged\n",
                encoding="utf-8",
            )
            disk_usage = mock.Mock(free=64 * 1024**3)

            with (
                mock.patch(
                    "assetforge.mflux_training._physical_memory_bytes",
                    return_value=32 * 1024**3,
                ),
                mock.patch(
                    "assetforge.mflux_training.shutil.disk_usage",
                    return_value=disk_usage,
                ),
                self.assertRaisesRegex(RuntimeError, "model changed.*fingerprint differs"),
            ):
                compile_mflux_train_command(plan, execute=True)

    def test_manifest_plan_locks_and_revalidates_model_weight_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = _model(root / "model")
            plan = build_mflux_training_plan(
                _dataset(root / "dataset"),
                model_path=model,
                config_output=root / "run" / "train.json",
                executable=_versioned_executable(root),
                physical_memory_bytes=32 * 1024**3,
                minimum_free_disk_gib=1.0,
            )
            self.assertTrue(plan["modelFingerprintVerified"])
            self.assertIsInstance(plan["modelLock"], dict)
            write_mflux_training_config(plan)

            weight = model / "transformer" / "0.safetensors"
            changed = bytearray(weight.read_bytes())
            changed[-1] ^= 1
            weight.write_bytes(changed)
            disk_usage = mock.Mock(free=64 * 1024**3)

            with (
                mock.patch(
                    "assetforge.mflux_training._physical_memory_bytes",
                    return_value=32 * 1024**3,
                ),
                mock.patch(
                    "assetforge.mflux_training.shutil.disk_usage",
                    return_value=disk_usage,
                ),
                self.assertRaisesRegex(RuntimeError, "model changed.*fingerprint differs"),
            ):
                compile_mflux_train_command(plan, execute=True)

    def test_model_validation_requires_every_indexed_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ready = validate_local_mflux_model(_model(root / "complete", indexed=True))
            self.assertTrue(ready["ready"])
            self.assertEqual(ready["indexedShardCount"], 4)

            with self.assertRaisesRegex(ValueError, "missing shard"):
                validate_local_mflux_model(_model(root / "missing", indexed=True, missing_shard=True))

    def test_training_entrypoints_reject_a_stored_four_bit_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            quantized = _model(root / "model-4bit", quantization_level=4)
            manifest = _dataset(root / "dataset")
            executable = _versioned_executable(root)

            report = validate_local_mflux_model(quantized)
            self.assertEqual(report["quantizationLevel"], 4)
            with self.assertRaisesRegex(ValueError, "requires unquantized base weights"):
                validate_local_mflux_model(
                    quantized,
                    require_unquantized=True,
                )
            with self.assertRaisesRegex(ValueError, "requires unquantized base weights"):
                create_portable_training_bundle(
                    manifest,
                    root / "portable",
                    model_path=quantized,
                )
            with self.assertRaisesRegex(ValueError, "requires unquantized base weights"):
                mflux_training_doctor(
                    model_path=quantized,
                    executable=executable,
                    physical_memory_bytes=32 * 1024**3,
                    minimum_free_disk_gib=1.0,
                )
            with self.assertRaisesRegex(ValueError, "requires unquantized base weights"):
                build_mflux_training_plan(
                    manifest,
                    model_path=quantized,
                    config_output=root / "run" / "train.json",
                    executable=executable,
                    physical_memory_bytes=32 * 1024**3,
                    minimum_free_disk_gib=1.0,
                )

    def test_model_validation_rejects_a_symlinked_model_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = _model(root / "model")
            alias = root / "model-alias"
            alias.symlink_to(model, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "model directory.*symbolic link"):
                validate_local_mflux_model(alias)

    def test_executable_discovery_is_authoritative_and_does_not_run_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = _executable(root / "mflux-train")

            self.assertEqual(
                discover_mflux_train_executable(executable, environ={"PATH": ""}),
                executable.resolve(),
            )
            self.assertIsNone(
                discover_mflux_train_executable(root / "missing", environ={"PATH": str(root)})
            )

    def test_doctor_blocks_local_training_below_24_gib(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = _model(root / "model")
            executable = _executable(root / "mflux-train")

            result = mflux_training_doctor(
                model_path=model,
                executable=executable,
                environ={"PATH": ""},
                physical_memory_bytes=16 * 1024**3,
            )

            self.assertFalse(result["localTrainingReady"])
            self.assertEqual(result["minimumPhysicalMemoryGiB"], 24.0)
            self.assertTrue(any("24 GiB" in blocker for blocker in result["blockers"]))
            self.assertTrue(any("version could not be detected" in warning for warning in result["warnings"]))
            self.assertTrue(result["trainingExecutionProvided"])
            self.assertTrue(result["trainingExecutionRequiresExplicitExecute"])

    def test_doctor_detects_exact_adjacent_dist_info_and_checks_training_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = _model(root / "model")
            executable = _versioned_executable(root)
            data = root / "data"
            data.mkdir()

            result = mflux_training_doctor(
                model_path=model,
                executable=executable,
                environ={"PATH": ""},
                physical_memory_bytes=32 * 1024**3,
                data_path=data,
                checkpoint_path=root / "run" / "checkpoints",
                minimum_free_disk_gib=1.0,
            )

            self.assertTrue(result["localTrainingReady"])
            self.assertEqual(result["mfluxVersion"]["detected"], MFLUX_TRAINING_VERSION)
            self.assertTrue(result["mfluxVersion"]["compatible"])
            self.assertTrue(result["paths"]["data"]["writable"])
            self.assertTrue(result["paths"]["checkpoint"]["writable"])
            self.assertEqual(
                Path(result["paths"]["cache"]["path"]),
                (data / ".mflux_cache" / "training").resolve(),
            )
            self.assertTrue(result["paths"]["cache"]["enoughFreeDisk"])

    def test_doctor_blocks_wrong_adjacent_version_and_low_free_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = _model(root / "model")
            executable = _versioned_executable(root, "0.17.0")
            data = root / "data"
            data.mkdir()
            disk_usage = mock.Mock(free=2 * 1024**3)

            with mock.patch("assetforge.mflux_training.shutil.disk_usage", return_value=disk_usage):
                result = mflux_training_doctor(
                    model_path=model,
                    executable=executable,
                    environ={"PATH": ""},
                    physical_memory_bytes=32 * 1024**3,
                    data_path=data,
                    checkpoint_path=root / "run" / "checkpoints",
                    minimum_free_disk_gib=MIN_LOCAL_TRAINING_FREE_DISK_GIB,
                )

            self.assertFalse(result["localTrainingReady"])
            self.assertFalse(result["mfluxVersion"]["compatible"])
            self.assertTrue(any("version mismatch" in blocker for blocker in result["blockers"]))
            disk_blockers = [
                blocker for blocker in result["blockers"] if "insufficient free disk" in blocker
            ]
            self.assertEqual(len(disk_blockers), 1)
            self.assertIn("data, checkpoint, cache", disk_blockers[0])

    def test_doctor_blocks_non_directory_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = _model(root / "model")
            executable = _versioned_executable(root)
            data = root / "data"
            data.mkdir()
            checkpoint = root / "checkpoint-file"
            checkpoint.write_text("not a directory\n", encoding="utf-8")

            result = mflux_training_doctor(
                model_path=model,
                executable=executable,
                physical_memory_bytes=32 * 1024**3,
                data_path=data,
                checkpoint_path=checkpoint,
                minimum_free_disk_gib=1.0,
            )

            self.assertFalse(result["localTrainingReady"])
            self.assertFalse(result["paths"]["checkpoint"]["ready"])
            self.assertTrue(any("must be a directory" in blocker for blocker in result["blockers"]))

    def test_doctor_blocks_managed_cache_symlink_that_escapes_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = _model(root / "model")
            executable = _versioned_executable(root)
            data = root / "data"
            data.mkdir()
            victim = root / "victim"
            (victim / "training").mkdir(parents=True)
            marker = victim / "training" / "KEEP"
            marker.write_text("preserve\n", encoding="utf-8")
            (data / ".mflux_cache").symlink_to(victim, target_is_directory=True)

            result = mflux_training_doctor(
                model_path=model,
                executable=executable,
                physical_memory_bytes=32 * 1024**3,
                data_path=data,
                checkpoint_path=root / "run" / "checkpoints",
                minimum_free_disk_gib=1.0,
            )

            self.assertFalse(result["localTrainingReady"])
            self.assertFalse(result["paths"]["cache"]["ready"])
            self.assertTrue(
                any("managed cache path" in blocker for blocker in result["blockers"])
            )
            self.assertTrue(marker.is_file())

    def test_builds_exact_0180_config_and_dry_run_argv_with_execute_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset_root = root / "dataset"
            manifest_path = _dataset(dataset_root, train_count=2, holdout_count=1)
            run_root = root / "run"
            prepared = prepare_training_data(manifest_path, run_root, sample_limit=1)
            model = _model(root / "model")
            executable = _versioned_executable(root)
            config_output = run_root / "train.json"

            plan = build_mflux_training_plan(
                manifest_path,
                model_path=model,
                config_output=config_output,
                prepared_data_path=prepared["dataPath"],
                sample_limit=1,
                executable=executable,
                environ={"PATH": ""},
                epochs=1,
                checkpoint_frequency=1,
                generate_image_frequency=1,
                physical_memory_bytes=32 * 1024**3,
                minimum_free_disk_gib=1.0,
            )

            self.assertTrue(plan["ready"])
            self.assertEqual(plan["mfluxVersion"], MFLUX_TRAINING_VERSION)
            self.assertEqual(plan["config"]["model"], MFLUX_TRAINING_MODEL)
            self.assertEqual(plan["config"]["model_path"], str(model.resolve()))
            self.assertEqual(plan["config"]["data"], prepared["dataPath"])
            self.assertFalse(plan["config"]["low_ram"])
            self.assertIsNone(plan["config"]["quantize"])
            self.assertEqual(plan["config"]["max_resolution"], 576)
            self.assertEqual(plan["config"]["training_loop"]["num_epochs"], 1)
            self.assertEqual(plan["config"]["training_loop"]["batch_size"], 1)
            self.assertEqual(plan["config"]["optimizer"]["learning_rate"], 1e-4)
            self.assertEqual(len(plan["config"]["lora_layers"]["targets"]), 14)
            self.assertFalse(plan["dataset"]["holdoutIncluded"])
            self.assertFalse(plan["execution"]["shell"])
            self.assertTrue(any("smoke subset" in warning for warning in plan["warnings"]))
            self.assertEqual(plan["schedule"]["mode"], "epochs")
            self.assertIsNone(plan["schedule"]["requestedTargetUpdates"])
            self.assertEqual(plan["schedule"]["plannedUpdates"], 1)

            written = write_mflux_training_config(plan)
            dry_run = compile_mflux_train_command(plan)
            self.assertEqual(written, config_output.resolve())
            self.assertEqual(
                dry_run,
                [str(executable.resolve()), "--config", str(config_output.resolve()), "--dry-run"],
            )
            disk_usage = mock.Mock(free=64 * 1024**3)
            with (
                mock.patch(
                    "assetforge.mflux_training._physical_memory_bytes",
                    return_value=16 * 1024**3,
                ),
                mock.patch(
                    "assetforge.mflux_training.shutil.disk_usage",
                    return_value=disk_usage,
                ),
                self.assertRaisesRegex(RuntimeError, "unsafe host.*24 GiB"),
            ):
                compile_mflux_train_command(plan, execute=True)
            with (
                mock.patch(
                    "assetforge.mflux_training._physical_memory_bytes",
                    return_value=32 * 1024**3,
                ),
                mock.patch(
                    "assetforge.mflux_training.shutil.disk_usage",
                    return_value=disk_usage,
                ),
            ):
                training = compile_mflux_train_command(plan, execute=True)
            self.assertEqual(
                training,
                [str(executable.resolve()), "--config", str(config_output.resolve())],
            )
            Path(plan["config"]["checkpoint"]["output_path"]).mkdir()
            with self.assertRaisesRegex(RuntimeError, "checkpoint output already exists"):
                compile_mflux_train_command(plan, execute=True)
            with self.assertRaisesRegex(FileExistsError, "overwrite is disabled"):
                write_mflux_training_config(plan)

    def test_managed_execution_blocks_mflux_recursive_low_ram_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = build_mflux_training_plan(
                _dataset(root / "dataset"),
                model_path=_model(root / "model"),
                config_output=root / "run" / "train.json",
                executable=_versioned_executable(root),
                low_ram=True,
                physical_memory_bytes=32 * 1024**3,
                minimum_free_disk_gib=1.0,
            )

            self.assertFalse(plan["ready"])
            self.assertTrue(plan["config"]["low_ram"])
            self.assertTrue(
                any("recursive deletion" in blocker for blocker in plan["blockers"])
            )

    def test_cli_keeps_inference_low_ram_on_and_training_low_ram_off_by_default(self) -> None:
        inference = assetforge_parser().parse_args(
            ["mflux-redraw", "--input", "input.png", "--output", "output.png"]
        )
        training = assetforge_parser().parse_args(
            [
                "mflux-train-plan",
                "--manifest",
                "dataset.json",
                "--model-path",
                "model",
                "--config-output",
                "train.json",
            ]
        )

        self.assertTrue(inference.low_ram)
        self.assertFalse(training.low_ram)

    def test_execution_rejects_checkpoint_ancestor_replaced_by_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            future = root / "future"
            plan = build_mflux_training_plan(
                _dataset(dataset),
                model_path=_model(root / "model"),
                config_output=root / "run" / "train.json",
                checkpoint_output=future / "artifacts",
                executable=_versioned_executable(root),
                physical_memory_bytes=32 * 1024**3,
                minimum_free_disk_gib=1.0,
            )
            self.assertTrue(plan["ready"])
            write_mflux_training_config(plan)
            future.symlink_to(dataset / "mflux" / "train", target_is_directory=True)

            with (
                mock.patch("assetforge.mflux_training.subprocess.run") as run,
                self.assertRaisesRegex(RuntimeError, "unsafe path.*symbolic link"),
            ):
                compile_mflux_train_command(plan, execute=True)
            run.assert_not_called()

    def test_training_command_requires_verified_exact_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root / "dataset")
            model = _model(root / "model")
            executable = _executable(root / "mflux-train")
            plan = build_mflux_training_plan(
                manifest_path,
                model_path=model,
                config_output=root / "run" / "train.json",
                executable=executable,
                environ={"PATH": ""},
                physical_memory_bytes=32 * 1024**3,
                minimum_free_disk_gib=1.0,
            )
            write_mflux_training_config(plan)
            disk_usage = mock.Mock(free=64 * 1024**3)

            with (
                mock.patch(
                    "assetforge.mflux_training._physical_memory_bytes",
                    return_value=32 * 1024**3,
                ),
                mock.patch(
                    "assetforge.mflux_training.shutil.disk_usage",
                    return_value=disk_usage,
                ),
                self.assertRaisesRegex(RuntimeError, "verified MFLUX 0.18.0"),
            ):
                compile_mflux_train_command(plan, execute=True)

    def test_explicit_training_runs_dry_run_first_and_requires_a_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root / "dataset")
            model = _model(root / "model")
            executable = _versioned_executable(root)
            plan = build_mflux_training_plan(
                manifest_path,
                model_path=model,
                config_output=root / "run" / "train.json",
                checkpoint_output=root / "run" / "artifacts",
                executable=executable,
                environ={"PATH": ""},
                epochs=1,
                checkpoint_frequency=1,
                physical_memory_bytes=32 * 1024**3,
                minimum_free_disk_gib=1.0,
            )
            write_mflux_training_config(plan)
            checkpoint = (
                Path(plan["config"]["checkpoint"]["output_path"])
                / "checkpoints"
                / "0000002_checkpoint.zip"
            )
            calls = 0

            def fake_run(argv, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    self.assertEqual(argv[-1], "--dry-run")
                    return mock.Mock(returncode=0, stdout="validated", stderr="")
                self.assertNotIn("--dry-run", argv)
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"checkpoint")
                return mock.Mock(returncode=0)

            disk_usage = mock.Mock(free=64 * 1024**3)
            runtime_environment = {"PATH": "", "CUDA_VISIBLE_DEVICES": "0"}
            with (
                mock.patch(
                    "assetforge.mflux_training._physical_memory_bytes",
                    return_value=32 * 1024**3,
                ),
                mock.patch(
                    "assetforge.mflux_training.shutil.disk_usage",
                    return_value=disk_usage,
                ),
                mock.patch(
                    "assetforge.mflux_training.subprocess.run",
                    side_effect=fake_run,
                ) as run,
                mock.patch(
                    "assetforge.mflux_training.mflux_training_doctor",
                    wraps=mflux_training_doctor,
                ) as doctor,
            ):
                result = run_mflux_training_plan(
                    plan,
                    execute=True,
                    environ=runtime_environment,
                )

            self.assertEqual(run.call_count, 2)
            self.assertEqual(result["latestCheckpoint"], str(checkpoint.resolve()))
            self.assertTrue(result["offlineModelMode"])
            for call in run.call_args_list:
                self.assertFalse(call.kwargs["shell"])
                self.assertEqual(call.kwargs["env"]["HF_HUB_OFFLINE"], "1")
                self.assertEqual(call.kwargs["env"]["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(doctor.call_count, 2)
            for call in doctor.call_args_list:
                self.assertEqual(call.kwargs["environ"]["PATH"], "")
                self.assertEqual(call.kwargs["environ"]["CUDA_VISIBLE_DEVICES"], "0")
                self.assertEqual(call.kwargs["environ"]["HF_HUB_OFFLINE"], "1")
                self.assertEqual(call.kwargs["environ"]["TRANSFORMERS_OFFLINE"], "1")

            with self.assertRaisesRegex(RuntimeError, "execute=True"):
                run_mflux_training_plan(plan)

    def test_explicit_training_never_runs_unverified_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root / "dataset")
            plan = build_mflux_training_plan(
                manifest_path,
                model_path=_model(root / "model"),
                config_output=root / "run" / "train.json",
                executable=_executable(root / "unverified" / "bin" / "mflux-train"),
                environ={"PATH": ""},
                physical_memory_bytes=32 * 1024**3,
                minimum_free_disk_gib=1.0,
            )
            write_mflux_training_config(plan)

            with (
                mock.patch("assetforge.mflux_training.subprocess.run") as run,
                self.assertRaisesRegex(RuntimeError, "verified MFLUX 0.18.0"),
            ):
                run_mflux_training_plan(plan, execute=True, environ={"PATH": ""})

            run.assert_not_called()

    def test_explicit_training_rechecks_cache_symlink_before_any_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root / "dataset")
            plan = build_mflux_training_plan(
                manifest_path,
                model_path=_model(root / "model"),
                config_output=root / "run" / "train.json",
                executable=_versioned_executable(root),
                environ={"PATH": ""},
                physical_memory_bytes=32 * 1024**3,
                minimum_free_disk_gib=1.0,
            )
            write_mflux_training_config(plan)
            data = Path(plan["config"]["data"])
            victim = root / "victim"
            (victim / "training").mkdir(parents=True)
            marker = victim / "training" / "KEEP"
            marker.write_text("preserve\n", encoding="utf-8")
            (data / ".mflux_cache").symlink_to(victim, target_is_directory=True)
            disk_usage = mock.Mock(free=64 * 1024**3)

            with (
                mock.patch(
                    "assetforge.mflux_training._physical_memory_bytes",
                    return_value=32 * 1024**3,
                ),
                mock.patch(
                    "assetforge.mflux_training.shutil.disk_usage",
                    return_value=disk_usage,
                ),
                mock.patch("assetforge.mflux_training.subprocess.run") as run,
                self.assertRaisesRegex(RuntimeError, "unsafe host.*managed cache path"),
            ):
                run_mflux_training_plan(plan, execute=True, environ={"PATH": ""})

            run.assert_not_called()
            self.assertTrue(marker.is_file())

    def test_plan_rejects_checkpoint_schedule_that_cannot_save(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "would produce no checkpoint"):
                build_mflux_training_plan(
                    _dataset(root / "dataset"),
                    model_path=_model(root / "model"),
                    config_output=root / "run" / "train.json",
                    executable=_versioned_executable(root),
                    epochs=1,
                    batch_size=2,
                    checkpoint_frequency=2,
                    physical_memory_bytes=32 * 1024**3,
                    minimum_free_disk_gib=1.0,
                )

    def test_default_schedule_derives_epochs_from_target_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = build_mflux_training_plan(
                _dataset(root / "dataset", train_count=32),
                model_path=_model(root / "model"),
                config_output=root / "run" / "train.json",
                executable=_versioned_executable(root),
                physical_memory_bytes=32 * 1024**3,
                minimum_free_disk_gib=1.0,
            )

            self.assertEqual(plan["schedule"]["mode"], "target-updates")
            self.assertEqual(plan["schedule"]["requestedTargetUpdates"], 1500)
            self.assertEqual(plan["schedule"]["updatesPerEpoch"], 32)
            self.assertEqual(plan["schedule"]["numEpochs"], 47)
            self.assertEqual(plan["schedule"]["plannedUpdates"], 1504)
            self.assertEqual(plan["schedule"]["targetOvershoot"], 4)
            self.assertEqual(plan["config"]["training_loop"]["num_epochs"], 47)
            self.assertEqual(plan["config"]["checkpoint"]["save_frequency"], 250)
            self.assertEqual(plan["config"]["monitoring"]["plot_frequency"], 25)
            self.assertEqual(plan["config"]["monitoring"]["generate_image_frequency"], 250)

    def test_target_updates_round_up_for_batch_size_and_reject_mixed_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            arguments = {
                "model_path": _model(root / "model"),
                "config_output": root / "run" / "train.json",
                "executable": _versioned_executable(root),
                "physical_memory_bytes": 32 * 1024**3,
                "minimum_free_disk_gib": 1.0,
            }
            manifest = _dataset(root / "dataset", train_count=5)
            plan = build_mflux_training_plan(
                manifest,
                target_updates=10,
                batch_size=2,
                checkpoint_frequency=3,
                **arguments,
            )

            self.assertEqual(plan["schedule"]["updatesPerEpoch"], 3)
            self.assertEqual(plan["schedule"]["numEpochs"], 4)
            self.assertEqual(plan["schedule"]["plannedUpdates"], 12)
            self.assertEqual(plan["schedule"]["targetOvershoot"], 2)
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                build_mflux_training_plan(
                    manifest,
                    epochs=2,
                    target_updates=10,
                    checkpoint_frequency=1,
                    **arguments,
                )

    def test_execute_plan_can_reuse_only_an_exact_existing_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = _dataset(root / "dataset")
            model = _model(root / "model")
            executable = _versioned_executable(root)
            arguments = {
                "model_path": model,
                "config_output": root / "run" / "train.json",
                "checkpoint_output": root / "run" / "artifacts",
                "executable": executable,
                "environ": {"PATH": ""},
                "physical_memory_bytes": 32 * 1024**3,
                "minimum_free_disk_gib": 1.0,
            }
            first = build_mflux_training_plan(manifest, **arguments)
            write_mflux_training_config(first)

            reused = build_mflux_training_plan(
                manifest,
                **arguments,
                allow_existing_config=True,
            )
            self.assertTrue(reused["ready"])
            self.assertTrue(reused["configReused"])

            Path(arguments["config_output"]).write_text("{}\n", encoding="utf-8")
            changed = build_mflux_training_plan(
                manifest,
                **arguments,
                allow_existing_config=True,
            )
            self.assertFalse(changed["ready"])
            self.assertFalse(changed["configReused"])
            self.assertTrue(any("differs" in blocker for blocker in changed["blockers"]))

    def test_smoke_plan_requires_a_verified_prepared_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root / "dataset")
            model = _model(root / "model")

            with self.assertRaisesRegex(ValueError, "prepare_training_data"):
                build_mflux_training_plan(
                    manifest_path,
                    model_path=model,
                    config_output=root / "run" / "train.json",
                    sample_limit=1,
                    executable=root / "missing",
                    environ={"PATH": ""},
                )

    def test_plan_refuses_any_implicit_lanczos_resize_of_pixel_boards(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root / "dataset")
            model = _model(root / "model")
            executable = _executable(root / "mflux-train")

            with self.assertRaisesRegex(ValueError, "would resize"):
                build_mflux_training_plan(
                    manifest_path,
                    model_path=model,
                    config_output=root / "run" / "train.json",
                    executable=executable,
                    environ={"PATH": ""},
                    max_resolution=63,
                    physical_memory_bytes=32 * 1024**3,
                )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "divisible by 16"):
                build_mflux_training_plan(
                    _dataset(root / "dataset", size=(65, 64)),
                    model_path=_model(root / "model"),
                    config_output=root / "run" / "train.json",
                    executable=_versioned_executable(root),
                    physical_memory_bytes=32 * 1024**3,
                )

    def test_command_compiler_rejects_tampered_or_unwritten_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = _dataset(root / "dataset")
            model = _model(root / "model")
            executable = _executable(root / "mflux-train")
            plan = build_mflux_training_plan(
                manifest_path,
                model_path=model,
                config_output=root / "run" / "train.json",
                executable=executable,
                environ={"PATH": ""},
            )

            with self.assertRaisesRegex(RuntimeError, "write and verify"):
                compile_mflux_train_command(plan)
            written = write_mflux_training_config(plan)
            written.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                compile_mflux_train_command(plan)


if __name__ == "__main__":
    unittest.main()
