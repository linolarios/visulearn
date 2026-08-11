"""Generate config/schemas/script_schema.json FROM the Pydantic models.

Run this whenever models.py changes:

    python scripts/generate_schema.py

The emitted schema is what you hand to a provider's structured-output mode
(Groq response_format json_schema, Ollama `format`, Gemini responseSchema).
Never hand-edit the JSON — edit models.py and regenerate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from repo root or scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import canonical_schema  # noqa: E402


def main() -> None:
    # models.canonical_schema() is the single definition — the Script Engine calls the
    # same function, so this file and the wire schema cannot drift apart.
    schema = canonical_schema()

    out = Path(__file__).resolve().parent.parent / "config" / "schemas" / "script_schema.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(json.dumps(schema))} bytes)")


if __name__ == "__main__":
    main()
