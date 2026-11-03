"""Versioned mock oral-image adapter. Research demo only — not for clinical use."""

from __future__ import annotations

import hashlib


MODEL_VERSION = "mock-v1"
PROVIDER = "mock"


def analyze(image_bytes: bytes, screening_id: str) -> dict:
    digest = hashlib.sha256(image_bytes).hexdigest()
    # Deterministic demo split so jury can see both labels without claiming accuracy.
    suspicious = int(digest[:2], 16) % 3 != 0
    label = "suspicious_example" if suspicious else "no_suspicious_finding_example"
    return {
        "provider": PROVIDER,
        "model_version": MODEL_VERSION,
        "research_label": label,
        "score": None,
        "clinical_use": False,
        "requires_clinician_review": True,
        "warning": "NOT FOR CLINICAL USE",
        "screening_id": screening_id,
    }
