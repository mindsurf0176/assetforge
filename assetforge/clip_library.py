"""Deterministic cutout animation clips for AssetForge rig archetypes.

The library contains motion data only: no project names, file paths, or art
assets. Translation channels are authored against :data:`BASE_HEIGHT` and are
scaled when a pack is requested. Rotation channels are always expressed in
degrees and are never scaled.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from numbers import Real
from typing import Any


BASE_HEIGHT = 256

_CLIP_ORDER = ("idle", "walk", "attack", "hit", "death")


_BIPED_SIDE: dict[str, dict[str, Any]] = {
    "idle": {
        "fps": 8,
        "loop": True,
        "tracks": {
            "root": {"offset_y": [[0, 0], [0.5, -1], [1, 0]]},
            "spine": {"rotation": [[0, 2], [0.5, 4.5], [1, 2]]},
            "neck": {"rotation": [[0, 0], [0.3, -1.5], [0.75, 1], [1, 0]]},
            "sh_f": {"rotation": [[0, 3], [0.5, 7], [1, 3]]},
            "sh_b": {"rotation": [[0, -2], [0.5, -6], [1, -2]]},
        },
    },
    "walk": {
        "fps": 10,
        "loop": True,
        "grounded": True,
        "tracks": {
            "root": {
                "offset_y": [[0, 0], [0.25, -2], [0.5, 0], [0.75, -2], [1, 0]],
            },
            "spine": {"rotation": [[0, 4], [0.25, 5.5], [0.5, 4], [0.75, 5.5], [1, 4]]},
            "neck": {"rotation": [[0, -1], [0.25, 1], [0.5, -1], [0.75, 1], [1, -1]]},
            "hip_f": {"rotation": [[0, 26], [0.5, -24], [1, 26]]},
            "kn_f": {"rotation": [[0, 6], [0.15, 42], [0.4, 6], [0.5, 2], [1, 6]]},
            "hip_b": {"rotation": [[0, -24], [0.5, 26], [1, -24]]},
            "kn_b": {"rotation": [[0, 2], [0.5, 6], [0.65, 42], [0.9, 6], [1, 2]]},
            "sh_f": {"rotation": [[0, -22], [0.5, 18], [1, -22]]},
            "el_f": {"rotation": [[0, -14], [0.5, -6], [1, -14]]},
            "sh_b": {"rotation": [[0, 18], [0.5, -22], [1, 18]]},
            "el_b": {"rotation": [[0, -6], [0.5, -14], [1, -6]]},
        },
    },
    "attack": {
        "fps": 14,
        "loop": False,
        "grounded": True,
        "tracks": {
            "root": {
                "offset_x": [[0, 0], [0.18, -9], [0.34, -11], [0.42, 17], [0.55, 21], [0.75, 6], [1, 0]],
                "offset_y": [[0, 0], [0.18, -3], [0.34, -3.5], [0.42, 5], [0.55, 6], [0.75, 1.5], [1, 0]],
            },
            "spine": {"rotation": [[0, 4], [0.18, -14], [0.34, -17], [0.42, 22], [0.52, 26], [0.72, 10], [1, 4]]},
            "neck": {"rotation": [[0, 0], [0.18, 8], [0.34, 9], [0.42, -6], [0.52, -8], [0.72, -2], [1, 0]]},
            "sh_f": {"rotation": [[0, -15], [0.18, -195], [0.34, -212], [0.42, -60], [0.52, -38], [0.7, -55], [1, -15]]},
            "el_f": {"rotation": [[0, -25], [0.18, -95], [0.34, -108], [0.42, -15], [0.52, -8], [0.7, -30], [1, -25]]},
            "sh_b": {"rotation": [[0, 10], [0.18, -38], [0.34, -45], [0.42, 42], [0.52, 50], [0.7, 20], [1, 10]]},
            "el_b": {"rotation": [[0, -20], [0.18, -55], [0.34, -60], [0.42, -12], [0.7, -25], [1, -20]]},
            "hip_f": {"rotation": [[0, -6], [0.18, 14], [0.34, 18], [0.42, -30], [0.55, -34], [0.75, -14], [1, -6]]},
            "kn_f": {"rotation": [[0, 6], [0.18, 10], [0.42, 18], [0.55, 22], [0.75, 10], [1, 6]]},
            "hip_b": {"rotation": [[0, 4], [0.18, -10], [0.34, -12], [0.42, 34], [0.55, 40], [0.75, 16], [1, 4]]},
            "kn_b": {"rotation": [[0, 4], [0.18, 14], [0.42, 8], [0.55, 6], [1, 4]]},
        },
    },
    "hit": {
        "fps": 14,
        "loop": False,
        "grounded": True,
        "tracks": {
            "root": {
                "offset_x": [[0, 0], [0.08, -13], [0.2, -11], [0.45, -4], [0.7, -1], [1, 0]],
                "offset_y": [[0, 0], [0.08, 2], [0.25, 0.75], [0.5, 0], [1, 0]],
            },
            "spine": {"rotation": [[0, 0], [0.08, -18], [0.22, -14], [0.5, -4], [0.75, 2], [1, 0]]},
            "neck": {"rotation": [[0, 0], [0.08, 14], [0.25, 10], [0.5, 2], [1, 0]]},
            "sh_f": {"rotation": [[0, -8], [0.08, -55], [0.25, -40], [0.6, -14], [1, -8]]},
            "el_f": {"rotation": [[0, -20], [0.08, -45], [0.25, -35], [0.6, -22], [1, -20]]},
            "sh_b": {"rotation": [[0, 6], [0.08, -35], [0.25, -25], [0.6, 0], [1, 6]]},
            "el_b": {"rotation": [[0, -15], [0.08, -38], [0.25, -30], [0.6, -18], [1, -15]]},
        },
    },
    "death": {
        "fps": 10,
        "loop": False,
        "grounded": True,
        "tracks": {
            "root": {
                "offset_x": [[0, 0], [0.12, -9], [0.3, -20], [0.55, -37], [0.7, -41], [1, -41]],
                "offset_y": [[0, 0], [0.12, 1.5], [0.3, 10], [0.55, 40], [0.66, 56], [0.74, 53], [0.82, 55], [1, 55]],
            },
            "spine": {"rotation": [[0, 0], [0.12, -14], [0.3, -30], [0.55, -66], [0.66, -74], [0.78, -70], [1, -70]]},
            "neck": {"rotation": [[0, 0], [0.12, 10], [0.3, 14], [0.55, 18], [0.66, 24], [0.78, 20], [1, 20]]},
            "hip_f": {"rotation": [[0, -2], [0.12, -10], [0.3, -26], [0.55, -58], [0.66, -70], [0.78, -66], [1, -66]]},
            "kn_f": {"rotation": [[0, 4], [0.12, 10], [0.3, 22], [0.55, 38], [0.66, 30], [1, 30]]},
            "hip_b": {"rotation": [[0, 2], [0.12, -4], [0.3, -16], [0.55, -44], [0.66, -56], [1, -54]]},
            "kn_b": {"rotation": [[0, 4], [0.12, 12], [0.3, 26], [0.55, 44], [0.66, 36], [1, 36]]},
            "sh_f": {"rotation": [[0, -6], [0.12, -70], [0.3, -95], [0.55, -60], [0.66, -30], [0.78, -38], [1, -38]]},
            "el_f": {"rotation": [[0, -18], [0.12, -50], [0.3, -40], [0.55, -15], [0.66, -8], [1, -8]]},
            "sh_b": {"rotation": [[0, 4], [0.12, -45], [0.3, -70], [0.55, -30], [0.66, -12], [1, -14]]},
            "el_b": {"rotation": [[0, -12], [0.12, -40], [0.3, -30], [0.55, -10], [1, -8]]},
        },
    },
}


_QUADRUPED_SIDE: dict[str, dict[str, Any]] = {
    "idle": {
        "fps": 8,
        "loop": True,
        "tracks": {
            "root": {"offset_y": [[0, 0], [0.5, -1], [1, 0]]},
            "body": {"rotation": [[0, 0], [0.5, 1.5], [1, 0]]},
            "neck": {"rotation": [[0, 1], [0.5, -2], [1, 1]]},
            "head": {"rotation": [[0, -1], [0.5, 2], [1, -1]]},
            "tail": {"rotation": [[0, -5], [0.5, 7], [1, -5]]},
            "wing_f": {"rotation": [[0, -3], [0.5, 3], [1, -3]]},
            "wing_b": {"rotation": [[0, -1], [0.5, 2], [1, -1]]},
        },
    },
    "walk": {
        "fps": 10,
        "loop": True,
        "grounded": True,
        "tracks": {
            "root": {
                "offset_y": [[0, 0], [0.25, -2], [0.5, 0], [0.75, -2], [1, 0]],
            },
            "body": {"rotation": [[0, 1.5], [0.25, -1.5], [0.5, 1.5], [0.75, -1.5], [1, 1.5]]},
            "neck": {"rotation": [[0, -2], [0.25, 2], [0.5, -2], [0.75, 2], [1, -2]]},
            "head": {"rotation": [[0, 1], [0.25, -1], [0.5, 1], [0.75, -1], [1, 1]]},
            "foreleg_f": {"rotation": [[0, 28], [0.5, -26], [1, 28]]},
            "foreleg_b": {"rotation": [[0, -26], [0.5, 28], [1, -26]]},
            "hindleg_f": {"rotation": [[0, -24], [0.5, 30], [1, -24]]},
            "hindleg_b": {"rotation": [[0, 30], [0.5, -24], [1, 30]]},
            "tail": {"rotation": [[0, -8], [0.25, 4], [0.5, -8], [0.75, 4], [1, -8]]},
            "wing_f": {"rotation": [[0, -5], [0.25, 3], [0.5, -5], [0.75, 3], [1, -5]]},
            "wing_b": {"rotation": [[0, -3], [0.25, 2], [0.5, -3], [0.75, 2], [1, -3]]},
        },
    },
    "attack": {
        "fps": 12,
        "loop": False,
        "grounded": True,
        "tracks": {
            "root": {
                "offset_x": [[0, 0], [0.25, -7], [0.42, 18], [0.55, 22], [0.8, 5], [1, 0]],
                "offset_y": [[0, 0], [0.25, 2], [0.42, -3], [0.55, -2], [1, 0]],
            },
            "body": {"rotation": [[0, 0], [0.25, -8], [0.42, 12], [0.58, 16], [0.8, 4], [1, 0]]},
            "neck": {"rotation": [[0, 2], [0.25, 16], [0.42, -20], [0.58, -26], [0.8, -5], [1, 2]]},
            "head": {"rotation": [[0, 0], [0.25, 10], [0.42, -18], [0.58, -22], [0.8, -4], [1, 0]]},
            "foreleg_f": {"rotation": [[0, 2], [0.25, -16], [0.42, 30], [0.58, 36], [0.8, 8], [1, 2]]},
            "foreleg_b": {"rotation": [[0, -2], [0.25, -12], [0.42, 22], [0.58, 28], [0.8, 5], [1, -2]]},
            "hindleg_f": {"rotation": [[0, 0], [0.25, 18], [0.42, -20], [0.58, -24], [0.8, -6], [1, 0]]},
            "hindleg_b": {"rotation": [[0, 0], [0.25, 14], [0.42, -16], [0.58, -20], [0.8, -4], [1, 0]]},
            "tail": {"rotation": [[0, -4], [0.25, 12], [0.5, -18], [0.72, -10], [1, -4]]},
            "wing_f": {"rotation": [[0, -4], [0.25, 18], [0.42, -24], [0.58, -30], [0.8, -8], [1, -4]]},
            "wing_b": {"rotation": [[0, -2], [0.25, 13], [0.42, -18], [0.58, -22], [0.8, -6], [1, -2]]},
        },
    },
    "hit": {
        "fps": 12,
        "loop": False,
        "grounded": True,
        "tracks": {
            "root": {
                "offset_x": [[0, 0], [0.08, -12], [0.22, -9], [0.5, -3], [1, 0]],
                "offset_y": [[0, 0], [0.08, 2], [0.3, 0.5], [1, 0]],
            },
            "body": {"rotation": [[0, 0], [0.08, -12], [0.25, -8], [0.55, 2], [1, 0]]},
            "neck": {"rotation": [[0, 0], [0.08, 16], [0.25, 10], [0.55, -2], [1, 0]]},
            "head": {"rotation": [[0, 0], [0.08, 12], [0.25, 7], [0.55, -1], [1, 0]]},
            "foreleg_f": {"rotation": [[0, 0], [0.08, -18], [0.3, -10], [1, 0]]},
            "foreleg_b": {"rotation": [[0, 0], [0.08, -14], [0.3, -8], [1, 0]]},
            "tail": {"rotation": [[0, -3], [0.12, 18], [0.35, 7], [0.65, -6], [1, -3]]},
            "wing_f": {"rotation": [[0, -3], [0.08, 18], [0.28, 8], [0.6, -6], [1, -3]]},
            "wing_b": {"rotation": [[0, -1], [0.08, 13], [0.28, 6], [0.6, -4], [1, -1]]},
        },
    },
    "death": {
        "fps": 10,
        "loop": False,
        "grounded": True,
        "tracks": {
            "root": {
                "offset_x": [[0, 0], [0.2, -8], [0.45, -23], [0.7, -34], [1, -34]],
                "offset_y": [[0, 0], [0.2, 4], [0.45, 20], [0.7, 42], [0.82, 39], [1, 40]],
            },
            "body": {"rotation": [[0, 0], [0.2, -8], [0.45, -32], [0.7, -72], [0.82, -67], [1, -68]]},
            "neck": {"rotation": [[0, 0], [0.2, 10], [0.45, 22], [0.7, 38], [1, 34]]},
            "head": {"rotation": [[0, 0], [0.2, 7], [0.45, 16], [0.7, 28], [1, 24]]},
            "foreleg_f": {"rotation": [[0, 0], [0.2, -12], [0.45, -38], [0.7, -58], [1, -55]]},
            "foreleg_b": {"rotation": [[0, 0], [0.2, -8], [0.45, -30], [0.7, -50], [1, -48]]},
            "hindleg_f": {"rotation": [[0, 0], [0.2, 14], [0.45, 38], [0.7, 62], [1, 58]]},
            "hindleg_b": {"rotation": [[0, 0], [0.2, 10], [0.45, 32], [0.7, 54], [1, 51]]},
            "tail": {"rotation": [[0, -4], [0.2, 6], [0.45, 18], [0.7, 30], [0.82, 24], [1, 25]]},
            "wing_f": {"rotation": [[0, -3], [0.2, 8], [0.45, 26], [0.7, 48], [0.82, 43], [1, 44]]},
            "wing_b": {"rotation": [[0, -1], [0.2, 6], [0.45, 20], [0.7, 40], [0.82, 36], [1, 37]]},
        },
    },
}


_PACKS = {
    "biped_side": _BIPED_SIDE,
    "quadruped_side": _QUADRUPED_SIDE,
}


def supported_archetypes() -> tuple[str, ...]:
    """Return archetype identifiers in stable display order."""

    return tuple(_PACKS)


def clip_names(archetype: str | None = None) -> tuple[str, ...]:
    """Return clip names for an archetype, or the common public clip set."""

    if archetype is None:
        return _CLIP_ORDER
    try:
        pack = _PACKS[archetype]
    except KeyError as exc:
        raise KeyError(
            f"unknown clip archetype {archetype!r}; available: {', '.join(supported_archetypes())}"
        ) from exc
    return tuple(name for name in _CLIP_ORDER if name in pack)


def _requested_names(archetype: str, requested: Sequence[str] | str | None) -> tuple[str, ...]:
    available = clip_names(archetype)
    if requested is None:
        return available
    if isinstance(requested, str):
        names = tuple(name.strip() for name in requested.split(",") if name.strip())
    else:
        names = tuple(requested)
    unknown = tuple(name for name in names if name not in available)
    if unknown:
        raise ValueError(
            f"unknown clips for {archetype}: {', '.join(unknown)}; available: {', '.join(available)}"
        )
    return tuple(dict.fromkeys(names))


def _scale_offsets(clip: dict[str, Any], scale: float) -> None:
    for channels in clip["tracks"].values():
        for channel in ("offset_x", "offset_y"):
            if channel in channels:
                channels[channel] = [
                    [time, round(value * scale, 6)] for time, value in channels[channel]
                ]


def clips_for(
    archetype: str,
    canvas_height: int | float,
    available_joints: Iterable[str],
    requested: Sequence[str] | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a scaled, joint-filtered independent copy of a clip pack.

    Only ``offset_x`` and ``offset_y`` values are scaled. Times, rotations,
    frame rates, and loop semantics remain identical to the authored pack.
    """

    if archetype not in _PACKS:
        raise KeyError(
            f"unknown clip archetype {archetype!r}; available: {', '.join(supported_archetypes())}"
        )
    if isinstance(canvas_height, bool) or not isinstance(canvas_height, Real) or canvas_height <= 0:
        raise ValueError("canvas_height must be a positive number")

    joints = {available_joints} if isinstance(available_joints, str) else set(available_joints)
    names = _requested_names(archetype, requested)
    scale = float(canvas_height) / BASE_HEIGHT
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        clip = deepcopy(_PACKS[archetype][name])
        _scale_offsets(clip, scale)
        clip["tracks"] = {
            joint: channels for joint, channels in clip["tracks"].items() if joint in joints
        }
        result[name] = clip
    return result


__all__ = ["BASE_HEIGHT", "clip_names", "clips_for", "supported_archetypes"]
