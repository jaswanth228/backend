"""Image quality checks for demonstration photos. Not a clinical tool."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO


MIN_BYTES = 1_500
MAX_BYTES = 8_000_000
MIN_WIDTH = 240
MIN_HEIGHT = 240


@dataclass
class QualityResult:
    status: str
    reason: str | None
    blur_score: float | None = None
    exposure: float | None = None


def check_quality(image_bytes: bytes, content_type: str | None) -> QualityResult:
    if not image_bytes or len(image_bytes) < MIN_BYTES:
        return QualityResult("recapture", "invalid_image")
    if len(image_bytes) > MAX_BYTES:
        return QualityResult("recapture", "file_too_large")
    if content_type and content_type.lower() not in {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }:
        return QualityResult("recapture", "unsupported_type")

    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return QualityResult("acceptable", None)

    try:
        img = Image.open(BytesIO(image_bytes))
        img = img.convert("RGB")
    except Exception:
        return QualityResult("recapture", "invalid_image")

    width, height = img.size
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return QualityResult("recapture", "too_small")

    arr = np.asarray(img)
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype("float32")
    exposure = float(gray.mean())
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    blur = float(gx.var() + gy.var())

    if blur < 80:
        return QualityResult("recapture", "blurry", blur_score=blur, exposure=exposure)
    if exposure < 35:
        return QualityResult("recapture", "underexposed", blur_score=blur, exposure=exposure)
    if exposure > 230:
        return QualityResult("recapture", "overexposed", blur_score=blur, exposure=exposure)
    return QualityResult("acceptable", None, blur_score=blur, exposure=exposure)
