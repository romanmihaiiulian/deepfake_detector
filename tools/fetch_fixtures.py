"""Download test fixtures into tests/.fixtures (gitignored, never committed).

Fixtures are public-domain NASA portraits from Wikimedia Commons, pinned by sha256. Tests
animate them into synthetic videos to exercise detection, tracking and alignment on real
faces. No face is ever manipulated: the lab's own tests use a fake swapper that only draws
an ellipse.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO / "tests" / ".fixtures"
USER_AGENT = "deepfake-detector-test-fixtures/0.1 (open-source test fixtures; python-urllib)"
RETRY_STATUS = {429, 502, 503, 504}


@dataclass(frozen=True, slots=True)
class Fixture:
    path: str
    url: str
    sha256: str
    license: str
    source: str


PORTRAITS: tuple[Fixture, ...] = (
    Fixture(
        "portraits/ellen_ochoa.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/e/e8/Ellen_Ochoa.jpg",
        "fad48e8e3d1a7e38fa328c2059040fcde95b0cc4b9f6647f48a91274bf8e7928",
        "Public domain (NASA)",
        "https://commons.wikimedia.org/wiki/File:Ellen_Ochoa.jpg",
    ),
    Fixture(
        "portraits/mae_jemison.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/5/55/Mae_Carol_Jemison.jpg",
        "7f9c8203cbf9e1304e30eedc2577d245818790f74cb817c45b17fecf7f188776",
        "Public domain (NASA)",
        "https://commons.wikimedia.org/wiki/File:Mae_Carol_Jemison.jpg",
    ),
    Fixture(
        "portraits/neil_armstrong.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/0/0d/Neil_Armstrong_pose.jpg",
        "2acf38e1ed3f9502daedb95cecdd4f69f1626f62241e4ce42a2ba3979b93dee0",
        "Public domain (NASA)",
        "https://commons.wikimedia.org/wiki/File:Neil_Armstrong_pose.jpg",
    ),
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, attempts: int = 5) -> None:
    """Download with backoff on rate limiting (Wikimedia answers 429 to bursts)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            return
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_STATUS or attempt == attempts - 1:
                raise
            retry_after = error.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.isdigit() else 5.0 * 2**attempt
            print(f"  {error.code} from server, retrying in {delay:.0f} s", file=sys.stderr)
            time.sleep(min(delay, 120.0))


def fetch(fixture: Fixture, root: Path = FIXTURES_DIR) -> Path:
    dest = root / fixture.path
    if dest.is_file() and sha256_of(dest) == fixture.sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    _download(fixture.url, tmp)
    actual = sha256_of(tmp)
    if actual != fixture.sha256:
        tmp.unlink()
        raise RuntimeError(f"{fixture.url}: sha256 {actual}, expected {fixture.sha256}")
    tmp.replace(dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=FIXTURES_DIR)
    args = parser.parse_args(argv)
    failed = False
    for fixture in PORTRAITS:
        try:
            print(f"ok      {fetch(fixture, args.dir)}  ({fixture.license})")
        except (OSError, RuntimeError) as exc:
            failed = True
            print(f"failed  {fixture.path}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
