"""Quick check of your Ollama Cloud setup, outside the web app.

    cd backend
    python scripts/test_ollama.py path/to/photo.jpg

Uses the same adapter (official `ollama` client) and the same .env as the API, so if this works,
"Analyse photo with AI" works too. Prints the structured feedback or the exact failure reason.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.ollama_adapter import OllamaError, analyze  # noqa: E402
from app.config import settings  # noqa: E402


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else input("Please enter the path to the image: ").strip().strip('"'))
    if not path.is_file():
        print(f"File not found: {path}")
        return 2
    print(f"Model: {settings.OLLAMA_MODEL} | Host: {settings.OLLAMA_BASE_URL} | Key set: {bool(settings.OLLAMA_API_KEY.strip())}")
    try:
        result = analyze(path.read_bytes(), "cli-test", {"view": "unspecified"})
    except OllamaError as exc:
        print(f"FAILED: {exc}")
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
