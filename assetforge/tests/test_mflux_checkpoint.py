from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

from assetforge.mflux_checkpoint import extract_mflux_lora, extract_mflux_lora_adapter


def _safetensors(*, paired: bool = True, marker: bytes = b"abcdefgh") -> bytes:
    names = (
        ["transformer.layer.lora_A.weight", "transformer.layer.lora_B.weight"]
        if paired
        else ["transformer.layer.weight", "transformer.other.weight"]
    )
    header = {
        names[0]: {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        names[1]: {"dtype": "U8", "shape": [4], "data_offsets": [4, 8]},
        "__metadata__": {
            "mflux_version": "0.18.0",
            "model": "flux2-klein-base-4b",
        },
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    return len(encoded).to_bytes(8, "little") + encoded + marker[:8].ljust(8, b"0")


def _checkpoint_json(lora_adapter: str = "0000001_adapter.safetensors") -> bytes:
    return (
        json.dumps(
            {
                "metadata": {"number_of_training_data": 1},
                "files": {
                    "config": "0000001_config.json",
                    "optimizer": "0000001_optimizer.safetensors",
                    "lora_adapter": lora_adapter,
                    "iterator": "0000001_iterator.json",
                    "loss": "0000001_loss.json",
                },
            },
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _directory_checkpoint(root: Path, *, lora: bytes | None = None) -> Path:
    root.mkdir()
    (root / "checkpoint.json").write_bytes(_checkpoint_json())
    (root / "0000001_adapter.safetensors").write_bytes(lora or _safetensors())
    (root / "0000001_optimizer.safetensors").write_bytes(_safetensors(marker=b"optimizer"))
    return root


def _zip_checkpoint(
    path: Path,
    *,
    lora: bytes | None = None,
    extra_members: list[tuple[str | zipfile.ZipInfo, bytes]] | None = None,
) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("checkpoint.json", _checkpoint_json())
        archive.writestr("0000001_adapter.safetensors", lora or _safetensors())
        archive.writestr("0000001_optimizer.safetensors", _safetensors(marker=b"optimizer"))
        for name, payload in extra_members or []:
            archive.writestr(name, payload)
    return path


class MfluxCheckpointTests(unittest.TestCase):
    def test_extracts_manifest_selected_lora_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = _directory_checkpoint(root / "checkpoint")
            output = root / "out" / "mongle.safetensors"

            report = extract_mflux_lora_adapter(checkpoint, output)

            source = checkpoint / "0000001_adapter.safetensors"
            self.assertEqual(output.read_bytes(), source.read_bytes())
            self.assertEqual(report["sourceType"], "directory")
            self.assertEqual(report["loraAdapter"], source.name)
            self.assertEqual(report["tensorCount"], 2)
            self.assertTrue(report["loraTensorPairsValid"])
            self.assertFalse(report["overwroteExisting"])
            self.assertEqual(report["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_extracts_manifest_selected_lora_from_zip_without_extractall(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = _zip_checkpoint(root / "checkpoint.zip")
            output = root / "lora.safetensors"

            report = extract_mflux_lora(checkpoint, output)

            self.assertEqual(output.read_bytes(), _safetensors())
            self.assertEqual(report["sourceType"], "zip")
            self.assertEqual(report["checkpointManifest"], "checkpoint.json")

    def test_follows_files_lora_adapter_instead_of_guessing_a_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            chosen = _safetensors(marker=b"selected")
            decoy = _safetensors(marker=b"decoy!!!")
            checkpoint = root / "checkpoint.zip"
            with zipfile.ZipFile(checkpoint, "w") as archive:
                archive.writestr("checkpoint.json", _checkpoint_json("chosen.safetensors"))
                archive.writestr("chosen.safetensors", chosen)
                archive.writestr("0000001_adapter.safetensors", decoy)
                archive.writestr("0000001_optimizer.safetensors", decoy)
            output = root / "result.safetensors"

            report = extract_mflux_lora(checkpoint, output)

            self.assertEqual(output.read_bytes(), chosen)
            self.assertEqual(report["loraAdapter"], "chosen.safetensors")

    def test_rejects_zip_slip_even_when_malicious_member_is_not_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = _zip_checkpoint(
                root / "checkpoint.zip",
                extra_members=[("../escaped.txt", b"malicious")],
            )
            output = root / "result.safetensors"

            with self.assertRaisesRegex(ValueError, "normalized relative path"):
                extract_mflux_lora(checkpoint, output)

            self.assertFalse(output.exists())
            self.assertFalse((root.parent / "escaped.txt").exists())

    def test_rejects_zip_symlink_even_when_it_is_not_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            symlink = zipfile.ZipInfo("unrelated-link")
            symlink.create_system = 3
            symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
            checkpoint = _zip_checkpoint(
                root / "checkpoint.zip",
                extra_members=[(symlink, b"/etc/passwd")],
            )
            output = root / "result.safetensors"

            with self.assertRaisesRegex(ValueError, "symbolic link"):
                extract_mflux_lora(checkpoint, output)

            self.assertFalse(output.exists())

    def test_rejects_duplicate_zip_members_and_duplicate_checkpoint_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "duplicate.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(checkpoint, "w") as archive:
                    archive.writestr("checkpoint.json", _checkpoint_json())
                    archive.writestr("0000001_adapter.safetensors", _safetensors())
                    archive.writestr("0000001_adapter.safetensors", _safetensors(marker=b"otherone"))
            with self.assertRaisesRegex(ValueError, "duplicate member"):
                extract_mflux_lora(checkpoint, root / "one.safetensors")

            nested = root / "nested-checkpoint.zip"
            with zipfile.ZipFile(nested, "w") as archive:
                archive.writestr("checkpoint.json", _checkpoint_json())
                archive.writestr("nested/checkpoint.json", _checkpoint_json())
                archive.writestr("0000001_adapter.safetensors", _safetensors())
            with self.assertRaisesRegex(ValueError, "exactly one root checkpoint.json"):
                extract_mflux_lora(nested, root / "two.safetensors")

    def test_rejects_manifest_traversal_and_missing_exact_member(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            traversal = root / "traversal.zip"
            with zipfile.ZipFile(traversal, "w") as archive:
                archive.writestr("checkpoint.json", _checkpoint_json("../adapter.safetensors"))
                archive.writestr("adapter.safetensors", _safetensors())
            with self.assertRaisesRegex(ValueError, "normalized relative path"):
                extract_mflux_lora(traversal, root / "one.safetensors")

            missing = root / "missing.zip"
            with zipfile.ZipFile(missing, "w") as archive:
                archive.writestr("checkpoint.json", _checkpoint_json("missing.safetensors"))
                archive.writestr("0000001_adapter.safetensors", _safetensors())
            with self.assertRaisesRegex(ValueError, "exactly one ZIP member"):
                extract_mflux_lora(missing, root / "two.safetensors")

    def test_rejects_disguised_safetensors_and_unpaired_tensors_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            disguised = _zip_checkpoint(root / "disguised.zip", lora=b"this is not safetensors")
            output = root / "disguised.safetensors"
            with self.assertRaisesRegex(ValueError, "safetensors"):
                extract_mflux_lora(disguised, output)
            self.assertFalse(output.exists())

            unpaired = _zip_checkpoint(root / "unpaired.zip", lora=_safetensors(paired=False))
            output = root / "unpaired.safetensors"
            with self.assertRaisesRegex(ValueError, "no matching LoRA tensor pair"):
                extract_mflux_lora(unpaired, output)
            self.assertFalse(output.exists())

    def test_rejects_lora_from_a_different_mflux_or_model_family(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wrong_version = _safetensors().replace(b'"0.18.0"', b'"0.17.0"')
            checkpoint = _zip_checkpoint(root / "wrong-version.zip", lora=wrong_version)
            with self.assertRaisesRegex(ValueError, "version does not match"):
                extract_mflux_lora(checkpoint, root / "version.safetensors")

            wrong_model = _safetensors().replace(
                b'"flux2-klein-base-4b"',
                b'"flux2-klein-base-9b"',
            )
            checkpoint = _zip_checkpoint(root / "wrong-model.zip", lora=wrong_model)
            with self.assertRaisesRegex(ValueError, "model does not identify"):
                extract_mflux_lora(checkpoint, root / "model.safetensors")

    def test_existing_output_is_never_modified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = _zip_checkpoint(root / "checkpoint.zip")
            output = root / "existing.safetensors"
            output.write_bytes(b"user-owned")

            with self.assertRaisesRegex(FileExistsError, "overwrite is disabled"):
                extract_mflux_lora(checkpoint, output)

            self.assertEqual(output.read_bytes(), b"user-owned")

    def test_rejects_directory_symlink_and_preserves_absent_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = _directory_checkpoint(root / "checkpoint")
            target = checkpoint / "real.safetensors"
            target.write_bytes(_safetensors())
            adapter = checkpoint / "0000001_adapter.safetensors"
            adapter.unlink()
            adapter.symlink_to(target.name)
            output = root / "result.safetensors"

            with self.assertRaisesRegex(ValueError, "symbolic link"):
                extract_mflux_lora(checkpoint, output)

            self.assertFalse(output.exists())

    def test_rejects_duplicate_checkpoint_json_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "duplicate-json.zip"
            manifest = b'{"files":{"lora_adapter":"a.safetensors","lora_adapter":"b.safetensors"}}'
            with zipfile.ZipFile(checkpoint, "w") as archive:
                archive.writestr("checkpoint.json", manifest)
                archive.writestr("a.safetensors", _safetensors())
                archive.writestr("b.safetensors", _safetensors())

            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                extract_mflux_lora(checkpoint, root / "result.safetensors")


if __name__ == "__main__":
    unittest.main()
