from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


PACKAGE_ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = PACKAGE_ROOT / "profiles" / "assetforge-profile.schema.json"
PROFILE_DIR = Path(
    os.environ.get("ASSETFORGE_PROFILE_DIR", PACKAGE_ROOT / "profiles")
).expanduser().resolve()


class ProfileError(ValueError):
    pass


@lru_cache(maxsize=1)
def _profile_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@dataclass(frozen=True)
class Profile:
    path: Path
    data: dict[str, Any]

    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def kind(self) -> str:
        return self.data["kind"]

    @property
    def project_root(self) -> Path:
        return Path(os.path.expandvars(self.data["projectRoot"])).expanduser().resolve()

    @property
    def fingerprint(self) -> str:
        stable = json.dumps(self.data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]

    def tier(self, name: str) -> dict[str, Any]:
        tiers = self.data.get("tiers", {})
        if name not in tiers:
            raise ProfileError(f"profile {self.id!r} has no tier {name!r}; choose one of {sorted(tiers)}")
        return tiers[name]

    def animation(self, name: str) -> dict[str, Any]:
        animations = self.data.get("animations", {})
        if name not in animations:
            raise ProfileError(
                f"profile {self.id!r} has no animation {name!r}; choose one of {sorted(animations)}"
            )
        return animations[name]


def _profile_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    named = PROFILE_DIR / (candidate.name if candidate.suffix else f"{candidate.name}.json")
    if named.is_file():
        return named.resolve()
    raise ProfileError(f"profile not found: {value}")


def validate_profile_data(data: dict[str, Any], source: str = "<memory>") -> None:
    errors = sorted(
        _profile_validator().iter_errors(data),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    location = ".".join(str(part) for part in error.absolute_path) or "<root>"
    raise ProfileError(f"{source}: {location}: {error.message}")


def load_profile(value: str | Path) -> Profile:
    path = _profile_path(value)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileError(f"{path}: invalid JSON: {exc}") from exc
    validate_profile_data(data, str(path))
    return Profile(path=path, data=data)


def list_profiles() -> list[dict[str, str]]:
    profiles = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        if path.name.endswith(".schema.json"):
            continue
        try:
            profile = load_profile(path)
        except ProfileError:
            continue
        profiles.append({"id": profile.id, "kind": profile.kind, "path": str(path)})
    return profiles
