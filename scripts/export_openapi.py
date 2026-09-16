#!/usr/bin/env python
"""Export the OpenAPI document to ``openapi.json``.

Used by ``make openapi`` and by the ``openapi-drift`` CI job. Runs with the
offline providers so it never needs credentials.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "openapi.json"

# Deterministic, credential-free settings for spec generation only.
SPEC_ENV = {
    "JWT_SECRET_KEY": "openapi-export-placeholder-secret-value-32+",
    "LLM_PROVIDER": "fake",
    "EMBEDDING_PROVIDER": "fake",
    "VECTOR_STORE": "chroma",
    "ENABLE_DOCS": "true",
    "ENVIRONMENT": "local",
}


def generate() -> dict[str, object]:
    """Build the application and return its OpenAPI schema."""
    for key, value in SPEC_ENV.items():
        os.environ[key] = value
    sys.path.insert(0, str(ROOT))

    from agent_service.config import build_settings
    from agent_service.main import create_app

    app = create_app(build_settings(_env_file=None, **{}))
    schema: dict[str, object] = app.openapi()
    return schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the file on disk differs from the generated spec.",
    )
    args = parser.parse_args(argv)

    rendered = json.dumps(generate(), indent=2, sort_keys=True) + "\n"

    if args.check:
        if not args.output.is_file():
            print(f"{args.output} does not exist - run `make openapi`.", file=sys.stderr)
            return 1
        if args.output.read_text(encoding="utf-8") != rendered:
            print(
                f"{args.output} is out of date - run `make openapi` and commit the result.",
                file=sys.stderr,
            )
            return 1
        print(f"{args.output} is up to date.")
        return 0

    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
