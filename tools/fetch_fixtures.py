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

# GenD's own sample face crops (256 px, aligned by GenD's preprocessing). They are frames of
# FaceForensics++ and Celeb-DF v2, which are research-only datasets: fetched at test time
# for the fidelity test, never committed, never distributed.
GEND_SAMPLE_COMMIT = "387a42266dd385fbe8f7626c5d3aa03eea3bfaab"
GEND_SAMPLE_BASE = f"https://raw.githubusercontent.com/yermandy/GenD/{GEND_SAMPLE_COMMIT}/datasets/"
GEND_SAMPLE_LICENSE = "Research-only (FaceForensics++ / Celeb-DF v2 terms)"
GEND_SAMPLE_SOURCE = f"https://github.com/yermandy/GenD/tree/{GEND_SAMPLE_COMMIT}/datasets"

GEND_SAMPLES: tuple[Fixture, ...] = (
    Fixture(
        "gend_samples/FF/DF/000_003/000.png",
        GEND_SAMPLE_BASE + "FF/DF/000_003/000.png",
        "b2c605732c6b2152320a986173c1a3dc7544938e948aa46444340831b8018060",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/FF/DF/000_003/012.png",
        GEND_SAMPLE_BASE + "FF/DF/000_003/012.png",
        "91b2bf24d9c3685c28559a5ac8c91b94bbe5b841a563b961f4a7679f40e8f4b8",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/FF/DF/000_003/025.png",
        GEND_SAMPLE_BASE + "FF/DF/000_003/025.png",
        "ca387094072caa412a0f683d909084580896f2c40dd669baf41c04166efffa95",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/FF/DF/000_003/038.png",
        GEND_SAMPLE_BASE + "FF/DF/000_003/038.png",
        "781d4ab6041c8b37c75b7d831f02944cae11b41fb848d5e34c577a142cb3a1a0",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/FF/F2F/000_003/000.png",
        GEND_SAMPLE_BASE + "FF/F2F/000_003/000.png",
        "a8e8c25ddc42909f3b82aedf939ca5fcf3924d1aeb54ad9dbb507ed97d050692",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/FF/FS/000_003/000.png",
        GEND_SAMPLE_BASE + "FF/FS/000_003/000.png",
        "57b282095be875d9b161360d43cbd4b04d0536c093bb967e6bc6661baf7db361",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/FF/NT/000_003/000.png",
        GEND_SAMPLE_BASE + "FF/NT/000_003/000.png",
        "ca2b6ce7ea30df50d86855adc4c2dd11b1ba0848153746303e73cb58d7035290",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/FF/real/000/000.png",
        GEND_SAMPLE_BASE + "FF/real/000/000.png",
        "33813fa2f7a716f20f27f11bf7e4126c53136108bec3db92dd6001e9453b185a",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/FF/real/000/012.png",
        GEND_SAMPLE_BASE + "FF/real/000/012.png",
        "d3af45388e764ae25e9a176a9ed148004d4e155146c9364199852a96815aafaf",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/FF/real/000/025.png",
        GEND_SAMPLE_BASE + "FF/real/000/025.png",
        "2b027fe48cf3468ad596acf3f8f9db59374dce243f57d4abdb89ab557e6681b6",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/FF/real/000/038.png",
        GEND_SAMPLE_BASE + "FF/real/000/038.png",
        "fc521c3b01c76428d353d734fde85969613e450666135881029c1e4e5cd38ce1",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/CDFv2/Celeb-real/id0_0000/000.png",
        GEND_SAMPLE_BASE + "CDFv2/Celeb-real/id0_0000/000.png",
        "33a652cb6ad545d41465a978ab4bb02137380db1747a1a460d193a3e0ecd4db6",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/CDFv2/Celeb-real/id0_0000/015.png",
        GEND_SAMPLE_BASE + "CDFv2/Celeb-real/id0_0000/015.png",
        "cca6e95d080ccafbd35709c0e6ce60a12b415c7f97cf40ac4c1edb7fba441e4f",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/CDFv2/Celeb-real/id0_0000/030.png",
        GEND_SAMPLE_BASE + "CDFv2/Celeb-real/id0_0000/030.png",
        "20d6775e831ef3bab80cd928c930b0916f09b933814fe0d2b1d6e51b0106c77a",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/CDFv2/Celeb-real/id0_0000/045.png",
        GEND_SAMPLE_BASE + "CDFv2/Celeb-real/id0_0000/045.png",
        "6ca1683cfe93a01ac7800de0efa5f82c38abb5b17582c5708671de567958e08e",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/CDFv2/Celeb-synthesis/id0_id1_0000/000.png",
        GEND_SAMPLE_BASE + "CDFv2/Celeb-synthesis/id0_id1_0000/000.png",
        "cae26283688b2e1855b75b922f75e0945dd29f1669e5c399b9e0f5bc75a4700c",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/CDFv2/Celeb-synthesis/id0_id1_0000/015.png",
        GEND_SAMPLE_BASE + "CDFv2/Celeb-synthesis/id0_id1_0000/015.png",
        "708f825b0fa8403e5db2966d96e4be4b18f4d1f55ab7352acdc2bd2d7542ee5b",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/CDFv2/Celeb-synthesis/id0_id1_0000/030.png",
        GEND_SAMPLE_BASE + "CDFv2/Celeb-synthesis/id0_id1_0000/030.png",
        "5693e8c8469630a8699f6cc4b8a51edd4297d31a956a14deba1a1b7796230ec4",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/CDFv2/Celeb-synthesis/id0_id1_0000/045.png",
        GEND_SAMPLE_BASE + "CDFv2/Celeb-synthesis/id0_id1_0000/045.png",
        "90c2250095d7331e64caa42d04d25cac5881288bbfa7013e68302cbe758ade06",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
    Fixture(
        "gend_samples/CDFv2/YouTube-real/00000/000.png",
        GEND_SAMPLE_BASE + "CDFv2/YouTube-real/00000/000.png",
        "c959f785348832e39685a4918904b3a145ff475ce69eef55624708b423473218",
        GEND_SAMPLE_LICENSE,
        GEND_SAMPLE_SOURCE,
    ),
)


def is_fake_sample(fixture: Fixture) -> bool:
    """Ground-truth label of a GenD sample crop, from its dataset folder."""
    return any(f"/{part}/" in fixture.path for part in ("DF", "F2F", "FS", "NT", "Celeb-synthesis"))


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
    for fixture in PORTRAITS + GEND_SAMPLES:
        try:
            print(f"ok      {fetch(fixture, args.dir)}  ({fixture.license})")
        except (OSError, RuntimeError) as exc:
            failed = True
            print(f"failed  {fixture.path}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
