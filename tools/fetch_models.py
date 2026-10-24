"""Download the pinned model weights into $DFD_MODELS_DIR (default ./models).

Every file is checked against the sha256 pinned in deepfake_core.models before it is
moved into place. `--verify` only checks files that are already present.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from deepfake_core import models
from deepfake_core.errors import ModelIntegrityError, ModelMissingError


def _progress(name: str) -> Callable[[int, int], None]:
    last = [-1]

    def report(received: int, total: int) -> None:
        percent = received * 100 // max(total, 1)
        if percent != last[0] and (percent % 5 == 0 or received == total):
            last[0] = percent
            print(f"\r  {name}: {percent:3d}% ({received / 1e6:,.0f} MB)", end="", flush=True)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="check, don't download")
    parser.add_argument("--dir", help=f"models directory (default ${models.MODELS_DIR_ENV})")
    parser.add_argument(
        "--only", action="append", choices=[spec.key for spec in models.MODELS], default=None
    )
    args = parser.parse_args(argv)

    root = models.models_dir(args.dir)
    failed = False
    for spec in models.MODELS:
        if args.only and spec.key not in args.only:
            continue
        try:
            if args.verify:
                path = models.verify(spec, root)
                print(f"ok      {spec.key:6} {path}")
            else:
                print(
                    f"fetch   {spec.key:6} {spec.name} ({spec.size / 1e6:,.1f} MB, {spec.license})"
                )
                path = models.fetch(spec, root, progress=_progress(spec.key))
                print(f"\nok      {spec.key:6} {path}")
        except (ModelMissingError, ModelIntegrityError, OSError) as exc:
            failed = True
            print(f"\nfailed  {spec.key:6} {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
