#!/usr/bin/env python
"""Dev-only helper that mints a JWT signed with ``JWT_SECRET_KEY``.

Deliberately a script and **never** an API endpoint: the service only ever
verifies tokens, it does not issue them.

    python scripts/mint_token.py --sub alice@example.com \\
        --scopes "kb:hr:read kb:hr:write agent:chat"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys

import jwt

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Populate ``os.environ`` from ``.env`` without overriding real env vars."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sub", default="dev@example.com", help="Token subject.")
    parser.add_argument(
        "--scopes",
        default="kb:*:read kb:*:write kb:admin agent:chat",
        help="Space separated scopes.",
    )
    parser.add_argument("--roles", default="", help="Space separated roles.")
    parser.add_argument("--tenant", default=None, help="Value for the 'tenant_id' claim.")
    parser.add_argument("--ttl", type=int, default=3600, help="Lifetime in seconds.")
    parser.add_argument("--issuer", default=None, help="Override JWT_ISSUER.")
    parser.add_argument("--audience", default=None, help="Override JWT_AUDIENCE.")
    parser.add_argument(
        "--claim", action="append", default=[], metavar="KEY=VALUE", help="Extra claim."
    )
    parser.add_argument("--quiet", action="store_true", help="Print only the token.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    args = build_parser().parse_args(argv)

    secret = os.environ.get("JWT_SECRET_KEY", "")
    if len(secret) < 32:
        print(
            "JWT_SECRET_KEY is missing or shorter than 32 characters. "
            "Set it in .env or the environment.",
            file=sys.stderr,
        )
        return 2

    algorithm = os.environ.get("JWT_ALGORITHM", "HS256")
    issuer = args.issuer or os.environ.get("JWT_ISSUER") or None
    audience = args.audience or os.environ.get("JWT_AUDIENCE") or None

    now = dt.datetime.now(tz=dt.UTC)
    payload: dict[str, object] = {
        "sub": args.sub,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int((now + dt.timedelta(seconds=args.ttl)).timestamp()),
        "scopes": args.scopes.split(),
    }
    if args.roles:
        payload["roles"] = args.roles.split()
    if args.tenant:
        payload["tenant_id"] = args.tenant
    if issuer:
        payload["iss"] = issuer
    if audience:
        payload["aud"] = audience
    for item in args.claim:
        key, _, value = item.partition("=")
        payload[key.strip()] = value.strip()

    token = jwt.encode(payload, secret, algorithm=algorithm)
    if args.quiet:
        print(token)
    else:
        print(json.dumps(payload, indent=2), file=sys.stderr)
        print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
