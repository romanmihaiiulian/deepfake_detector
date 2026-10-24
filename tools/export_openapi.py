"""Write the API's OpenAPI schema; the web app generates its TypeScript types from it.

uv run python -m tools.export_openapi apps/web/openapi.json          # regenerate
uv run python -m tools.export_openapi apps/web/openapi.json --check  # fail if stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from deepfake_api.main import create_app
from deepfake_api.settings import Settings


def openapi_schema() -> dict[str, Any]:
    # Building the app touches no disk and loads no model; the lifespan never runs here.
    schema: dict[str, Any] = create_app(Settings(analyzer="fake", web_dist=None)).openapi()
    return schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true", help="fail if the file is out of date")
    args = parser.parse_args(argv)
    text = json.dumps(openapi_schema(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text() != text:
            print(f"{args.output} is out of date: run `make gen-api`", file=sys.stderr)
            return 1
        print(f"{args.output} is up to date")
        return 0
    args.output.write_text(text)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
