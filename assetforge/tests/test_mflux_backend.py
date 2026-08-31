from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from assetforge.mflux_backend import (
    compile_mflux_command,
    discover_mflux_executable,
    mflux_doctor,
    paired_board_edit_plan,
    run_mflux_plan,
)


def _write_safetensors(path: Path, names: tuple[str, ...] = ("weight",)) -> None:
    tensors = {}
    data = bytearray()
    for name in names:
        start = len(data)
        data.extend(b"\0\0\0\0")
        tensors[name] = {"dtype": "F32", "shape": [1], "data_offsets": [start, len(data)]}
    header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(len(header).to_bytes(8, "little") + header + data)


def _executable(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)


def _model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    for component in ("text_encoder", "transformer", "vae"):
        root = path / component
        root.mkdir()
        _write_safetensors(root / "0.safetensors")
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {f"{component}.weight": "0.safetensors"}}),
            encoding="utf-8",
        )
    tokenizer = path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")


class MfluxBackendTests(unittest.TestCase):
    def test_explicit_executable_is_authoritative_and_not_started(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "mflux edit"
            _executable(executable)

            discovered = discover_mflux_executable(executable, environ={"PATH": ""})

            self.assertEqual(discovered, executable.resolve())
            self.assertIsNone(discover_mflux_executable(root / "missing", environ={"PATH": ""}))

    def test_doctor_fails_closed_without_complete_model_or_lora(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "mflux-generate-flux2-edit"
            _executable(executable)
            cache = root / "cache"
            cache.mkdir()

            result = mflux_doctor(
                executable=executable,
                model_path=root / "partial-model",
                cache_dir=cache,
                lora=root / "missing.safetensors",
                disk_path=root,
                minimum_free_gib=0,
                environ={},
            )

            self.assertFalse(result["ok"])
            self.assertFalse(result["model"]["ready"])
            self.assertFalse(result["lora"]["ready"])
            self.assertTrue(any("model weights" in blocker for blocker in result["blockers"]))
            self.assertTrue(any("LoRA" in blocker for blocker in result["blockers"]))

    def test_doctor_rejects_model_with_missing_indexed_weight_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "mflux-generate-flux2-edit"
            _executable(executable)
            model = root / "flux2-klein-4b-incomplete"
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            _write_safetensors(model / "1.safetensors")
            (model / "model.safetensors.index.json").write_text(
                '{"weight_map":{"a":"0.safetensors","b":"1.safetensors"}}\n',
                encoding="utf-8",
            )
            cache = root / "cache"
            cache.mkdir()

            result = mflux_doctor(
                executable=executable,
                model_path=model,
                cache_dir=cache,
                disk_path=root,
                minimum_free_gib=0,
                environ={},
            )

            self.assertFalse(result["model"]["ready"])
            self.assertTrue(any("model weights" in blocker for blocker in result["blockers"]))

    def test_doctor_ignores_huggingface_local_dir_download_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "mflux-generate-flux2-edit"
            _executable(executable)
            model = root / "flux2-klein-4b-model"
            _model(model)
            metadata = model / ".cache" / "huggingface" / "download"
            metadata.mkdir(parents=True)
            (metadata / "stale.incomplete").write_bytes(b"partial-cache-data")
            cache = root / "cache"
            cache.mkdir()

            result = mflux_doctor(
                executable=executable,
                model_path=model,
                cache_dir=cache,
                disk_path=root,
                minimum_free_gib=0,
                environ={},
            )

            self.assertTrue(result["model"]["ready"])
            self.assertTrue(result["ok"])

    def test_doctor_uses_mflux_literal_alias_resolution_without_punctuation_folding(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "mflux-generate-flux2-edit"
            _executable(executable)
            model = root / "flux2klein4b-model"
            _model(model)
            cache = root / "cache"
            cache.mkdir()

            result = mflux_doctor(
                executable=executable,
                model_path=model,
                cache_dir=cache,
                disk_path=root,
                minimum_free_gib=0,
                environ={},
            )

            self.assertFalse(result["ok"])
            self.assertTrue(result["model"]["ready"])
            self.assertTrue(any("recognizable base-model alias" in item for item in result["blockers"]))

    def test_plan_compiles_deterministic_safe_argv_with_optional_lora(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "mflux-generate-flux2-edit"
            _executable(executable)
            model = root / "flux2-klein-base-4b local model"
            _model(model)
            cache = root / "cache"
            cache.mkdir()
            lora = root / "sprite edit.safetensors"
            _write_safetensors(
                lora,
                ("transformer.block.lora_A.weight", "transformer.block.lora_B.weight"),
            )
            input_path = root / "input board.png"
            Image.new("RGB", (64, 80), (236, 244, 241)).save(input_path)
            output_path = root / "output board.png"
            prompt = "keep identity; $(touch should-not-run)"

            first = paired_board_edit_plan(
                input_path,
                output_path,
                prompt=prompt,
                executable=executable,
                model="flux2-klein-base-4b",
                model_path=model,
                cache_dir=cache,
                lora=lora,
                lora_scale=0.75,
                seed=123,
                quantize=4,
                minimum_free_gib=0,
                environ={},
            )
            second = paired_board_edit_plan(
                input_path,
                output_path,
                prompt=prompt,
                executable=executable,
                model="flux2-klein-base-4b",
                model_path=model,
                cache_dir=cache,
                lora=lora,
                lora_scale=0.75,
                seed=123,
                quantize=4,
                minimum_free_gib=0,
                environ={},
            )

            argv = compile_mflux_command(first)
            self.assertTrue(first["ready"])
            self.assertFalse(first["execution"]["shell"])
            self.assertEqual(argv, compile_mflux_command(second))
            self.assertEqual(argv[0], str(executable.resolve()))
            self.assertEqual(argv[argv.index("--base-model") + 1], "flux2-klein-base-4b")
            self.assertEqual(argv[argv.index("--image-paths") + 1], str(input_path.resolve()))
            self.assertEqual(argv[argv.index("--output") + 1], str(output_path.resolve()))
            self.assertEqual(argv[argv.index("--prompt") + 1], prompt)
            self.assertEqual(argv[argv.index("--width") + 1], "64")
            self.assertEqual(argv[argv.index("--height") + 1], "80")
            self.assertEqual(argv[argv.index("--lora-paths") + 1], str(lora.resolve()))
            self.assertNotIn("sh", argv)
            self.assertNotIn("-c", argv)

    def test_blocked_plan_cannot_compile_and_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "mflux-generate-flux2-edit"
            _executable(executable)
            model = root / "flux2-klein-4b-model"
            _model(model)
            cache = root / "cache"
            cache.mkdir()
            input_path = root / "input.png"
            output_path = root / "output.png"
            Image.new("RGB", (64, 64)).save(input_path)
            Image.new("RGB", (64, 64)).save(output_path)

            plan = paired_board_edit_plan(
                input_path,
                output_path,
                prompt="--output",
                executable=executable,
                model_path=model,
                cache_dir=cache,
                minimum_free_gib=0,
                environ={},
            )

            self.assertFalse(plan["ready"])
            self.assertIsNone(plan["argv"])
            self.assertIn("overwrite is disabled", plan["output"]["error"])
            with self.assertRaisesRegex(RuntimeError, "refusing to compile"):
                compile_mflux_command(plan)

    def test_rejects_non_aligned_board_before_command_compilation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.png"
            Image.new("RGB", (65, 64)).save(input_path)

            plan = paired_board_edit_plan(
                input_path,
                root / "output.png",
                executable=root / "missing",
                model_path=root / "missing-model",
                cache_dir=root,
                minimum_free_gib=0,
                environ={},
            )

            self.assertFalse(plan["input"]["ready"])
            self.assertIn("divisible by 16", plan["input"]["error"])

    def test_compiles_quantized_runpod_model_and_multiple_loras_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "mflux-generate-flux2-edit"
            _executable(executable)
            cache = root / "cache"
            snapshot = (
                cache
                / "models--Runpod--FLUX.2-klein-4B-mflux-4bit"
                / "snapshots"
                / "snapshot-a"
            )
            _model(snapshot)
            lora_a = root / "identity.safetensors"
            lora_b = root / "pixel-style.safetensors"
            _write_safetensors(
                lora_a,
                ("transformer.identity.lora_A.weight", "transformer.identity.lora_B.weight"),
            )
            _write_safetensors(
                lora_b,
                ("transformer.style.lora_A.weight", "transformer.style.lora_B.weight"),
            )
            input_path = root / "input.png"
            Image.new("RGB", (64, 64)).save(input_path)

            plan = paired_board_edit_plan(
                input_path,
                root / "output.png",
                executable=executable,
                model="Runpod/FLUX.2-klein-4B-mflux-4bit",
                base_model="flux2-klein-4b",
                cache_dir=cache,
                lora=[lora_a, lora_b],
                lora_scale=[0.8, 0.55],
                low_ram=True,
                mlx_cache_limit_gib=3.5,
                minimum_free_gib=0,
                environ={},
            )

            argv = compile_mflux_command(plan)
            self.assertEqual(
                argv[:5],
                [
                    str(executable.resolve()),
                    "--model",
                    str(snapshot.resolve()),
                    "--base-model",
                    "flux2-klein-4b",
                ],
            )
            self.assertNotIn("--quantize", argv)
            self.assertIn("--low-ram", argv)
            self.assertEqual(argv[argv.index("--mlx-cache-limit-gb") + 1], "3.5")
            paths_index = argv.index("--lora-paths")
            scales_index = argv.index("--lora-scales")
            self.assertEqual(argv[paths_index + 1 : scales_index], [str(lora_a.resolve()), str(lora_b.resolve())])
            self.assertEqual(argv[scales_index + 1 : scales_index + 3], ["0.8", "0.55"])
            self.assertTrue(any("ignore --base-model" in warning for warning in plan["doctor"]["warnings"]))

    def test_default_steps_follow_resolved_base_model_not_unrelated_model_argument(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "mflux-generate-flux2-edit"
            _executable(executable)
            model = root / "flux2-klein-base-4b-local"
            _model(model)
            cache = root / "cache"
            cache.mkdir()
            input_path = root / "input.png"
            Image.new("RGB", (64, 64)).save(input_path)

            plan = paired_board_edit_plan(
                input_path,
                root / "output.png",
                executable=executable,
                model_path=model,
                base_model="flux2-klein-base-4b",
                cache_dir=cache,
                minimum_free_gib=0,
                environ={},
            )

            self.assertTrue(plan["ready"])
            self.assertEqual(plan["parameters"]["steps"], 50)
            argv = compile_mflux_command(plan)
            self.assertEqual(argv[argv.index("--steps") + 1], "50")

    def test_execution_requires_gate_and_verifies_generated_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "mflux-generate-flux2-edit"
            _executable(executable)
            model = root / "flux2-klein-4b-model"
            _model(model)
            cache = root / "cache"
            cache.mkdir()
            input_path = root / "input.png"
            output_path = root / "output.png"
            Image.new("RGB", (64, 64)).save(input_path)
            plan = paired_board_edit_plan(
                input_path,
                output_path,
                executable=executable,
                model_path=model,
                cache_dir=cache,
                minimum_free_gib=0,
                environ={},
            )

            with self.assertRaisesRegex(RuntimeError, "execute=True"):
                run_mflux_plan(plan)

            def fake_run(*args, **kwargs):
                self.assertFalse(kwargs["shell"])
                argv = args[0]
                self.assertEqual(argv[-2], "--output")
                staged_output = Path(argv[-1])
                self.assertNotEqual(staged_output, output_path)
                Image.new("RGB", (64, 64)).save(staged_output)
                return type("Completed", (), {"returncode": 0})()

            with patch("assetforge.mflux_backend.subprocess.run", side_effect=fake_run):
                result = run_mflux_plan(plan, execute=True)

            self.assertTrue(result["ok"])
            self.assertEqual(result["size"], [64, 64])

    def test_overwrite_stages_then_atomically_replaces_only_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "mflux-generate-flux2-edit"
            _executable(executable)
            model = root / "flux2-klein-4b-model"
            _model(model)
            cache = root / "cache"
            cache.mkdir()
            input_path = root / "input.png"
            output_path = root / "output.png"
            Image.new("RGB", (64, 64), "white").save(input_path)
            Image.new("RGB", (64, 64), "red").save(output_path)
            plan = paired_board_edit_plan(
                input_path,
                output_path,
                prompt="--output",
                executable=executable,
                model_path=model,
                cache_dir=cache,
                overwrite=True,
                minimum_free_gib=0,
                environ={},
            )

            def successful_run(*args, **kwargs):
                self.assertEqual(Image.open(output_path).getpixel((0, 0)), (255, 0, 0))
                argv = args[0]
                self.assertEqual(argv[-2], "--output")
                staged_output = Path(argv[-1])
                Image.new("RGB", (64, 64), "blue").save(staged_output)
                staged_output.with_suffix(".metadata.json").write_text("{}\n", encoding="utf-8")
                return type("Completed", (), {"returncode": 0})()

            with patch("assetforge.mflux_backend.subprocess.run", side_effect=successful_run):
                run_mflux_plan(plan, execute=True)

            self.assertEqual(Image.open(output_path).getpixel((0, 0)), (0, 0, 255))
            self.assertTrue(output_path.with_suffix(".metadata.json").is_file())
            self.assertFalse(any(path.name.startswith(".assetforge-mflux-") for path in root.iterdir()))

            plan = paired_board_edit_plan(
                input_path,
                output_path,
                executable=executable,
                model_path=model,
                cache_dir=cache,
                overwrite=True,
                minimum_free_gib=0,
                environ={},
            )
            failed = type("Completed", (), {"returncode": 9})()
            with patch("assetforge.mflux_backend.subprocess.run", return_value=failed):
                with self.assertRaisesRegex(RuntimeError, "status 9"):
                    run_mflux_plan(plan, execute=True)
            self.assertEqual(Image.open(output_path).getpixel((0, 0)), (0, 0, 255))

    def test_metadata_publish_failure_restores_previous_output_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "mflux-generate-flux2-edit"
            _executable(executable)
            model = root / "flux2-klein-4b-model"
            _model(model)
            cache = root / "cache"
            cache.mkdir()
            input_path = root / "input.png"
            output_path = root / "output.png"
            metadata_path = output_path.with_suffix(".metadata.json")
            Image.new("RGB", (64, 64), "white").save(input_path)
            Image.new("RGB", (64, 64), "red").save(output_path)
            metadata_path.write_text('{"generation":"previous"}\n', encoding="utf-8")
            plan = paired_board_edit_plan(
                input_path,
                output_path,
                executable=executable,
                model_path=model,
                cache_dir=cache,
                overwrite=True,
                minimum_free_gib=0,
                environ={},
            )

            def successful_run(argv, **kwargs):
                staged_output = Path(argv[-1])
                Image.new("RGB", (64, 64), "blue").save(staged_output)
                staged_output.with_suffix(".metadata.json").write_text(
                    '{"generation":"new"}\n', encoding="utf-8"
                )
                return type("Completed", (), {"returncode": 0})()

            real_replace = os.replace
            injected = False

            def fail_metadata_publish(source, destination):
                nonlocal injected
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    not injected
                    and source_path.name == "output.metadata.json"
                    and destination_path.resolve() == metadata_path.resolve()
                ):
                    injected = True
                    raise OSError("injected metadata publish failure")
                real_replace(source, destination)

            with (
                patch("assetforge.mflux_backend.subprocess.run", side_effect=successful_run),
                patch("assetforge.mflux_backend.os.replace", side_effect=fail_metadata_publish),
                self.assertRaisesRegex(RuntimeError, "previous output restored"),
            ):
                run_mflux_plan(plan, execute=True)

            self.assertTrue(injected)
            self.assertEqual(Image.open(output_path).getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(
                metadata_path.read_text(encoding="utf-8"),
                '{"generation":"previous"}\n',
            )
            self.assertFalse(any(path.name.startswith(".assetforge-mflux-") for path in root.iterdir()))


if __name__ == "__main__":
    unittest.main()
