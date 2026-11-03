"""Supabase Storage client (REST via httpx). Images live in a private bucket; the DB keeps only the object key.

Needs SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY (server-side only, never ship to the browser).
"""
import httpx

from .config import settings

_TIMEOUT = 30.0


def _base() -> str:
    return settings.SUPABASE_URL.rstrip("/") + "/storage/v1"


def _headers(extra: dict | None = None) -> dict:
    key = settings.SUPABASE_SERVICE_ROLE_KEY
    h = {"Authorization": f"Bearer {key}", "apikey": key}
    if extra:
        h.update(extra)
    return h


def _require() -> None:
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in backend/.env")


def ensure_bucket() -> None:
    """Create the private bucket if it does not exist yet (idempotent)."""
    _require()
    r = httpx.post(
        f"{_base()}/bucket",
        headers=_headers({"Content-Type": "application/json"}),
        json={"id": settings.SUPABASE_BUCKET, "name": settings.SUPABASE_BUCKET, "public": False},
        timeout=_TIMEOUT,
    )
    if r.status_code in (200, 201):
        return
    # Already exists -> Supabase answers 400/409 with "Duplicate"/"already exists"
    body = r.text.lower()
    if r.status_code in (400, 409) and ("exist" in body or "duplicate" in body):
        return
    raise RuntimeError(f"Could not create Supabase bucket: {r.status_code} {r.text}")


def upload(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    _require()
    r = httpx.post(
        f"{_base()}/object/{settings.SUPABASE_BUCKET}/{key}",
        headers=_headers({"Content-Type": content_type, "x-upsert": "true"}),
        content=data,
        timeout=_TIMEOUT,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Supabase upload failed: {r.status_code} {r.text}")


def download(key: str) -> bytes | None:
    """Return the object's bytes, or None if it does not exist."""
    _require()
    r = httpx.get(
        f"{_base()}/object/authenticated/{settings.SUPABASE_BUCKET}/{key}",
        headers=_headers(),
        timeout=_TIMEOUT,
    )
    if r.status_code in (400, 404):
        return None
    if r.status_code != 200:
        raise RuntimeError(f"Supabase download failed: {r.status_code} {r.text}")
    return r.content


def detect_type(data: bytes) -> tuple[str, str]:
    """Return (extension, mime) from the file's magic bytes; falls back to .bin."""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg", "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png", "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp", "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif", "image/gif"
    return ".bin", "application/octet-stream"


MIME_BY_EXT = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}


def mime_for_key(key: str) -> str:
    return MIME_BY_EXT.get("." + key.rsplit(".", 1)[-1].lower(), "application/octet-stream")


def remove(key: str) -> None:
    _require()
    r = httpx.request(
        "DELETE",
        f"{_base()}/object/{settings.SUPABASE_BUCKET}",
        headers=_headers({"Content-Type": "application/json"}),
        json={"prefixes": [key]},
        timeout=_TIMEOUT,
    )
    if r.status_code not in (200, 204):
        raise RuntimeError(f"Supabase delete failed: {r.status_code} {r.text}")
