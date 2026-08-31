"""Optional vision backends for generated animation-sheet cleanup.

The core pipeline remains Pillow-only. These integrations are deliberately lazy:
installing AssetForge does not install OpenCV, torch, or segmentation weights.
"""

from __future__ import annotations

from typing import Any

from PIL import Image


def opencv_connected_components(mask: Any) -> tuple[int, Any, Any, Any]:
    """Run OpenCV connected-components when the optional extra is installed."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV backend is unavailable; install assetforge-2d[vision]"
        ) from exc
    return cv2.connectedComponentsWithStats(mask, connectivity=8)


def remove_background_rembg(
    image: Image.Image,
    *,
    model: str = "u2net",
    alpha_matting: bool = False,
) -> Image.Image:
    """Return a rembg cutout without making rembg a core dependency."""

    try:
        from rembg import new_session, remove
    except ImportError as exc:
        raise RuntimeError(
            "rembg backend is unavailable; install assetforge-2d[background]"
        ) from exc
    session = new_session(model)
    result = remove(
        image.convert("RGBA"),
        session=session,
        alpha_matting=alpha_matting,
        post_process_mask=True,
    )
    return result.convert("RGBA")


def sam2_status() -> dict[str, Any]:
    """Report SAM 2 availability without importing torch or loading weights."""

    try:
        import sam2  # type: ignore[import-not-found]
    except ImportError:
        return {
            "available": False,
            "reason": "install SAM 2 and its torch dependency separately",
        }
    return {"available": True, "module": str(sam2)}
