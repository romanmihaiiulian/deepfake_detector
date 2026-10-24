"""Temporary storage for uploads and evidence crops (see docs/privacy.md).

Uploads exist only while they are analysed: the worker deletes each one when its analysis
ends, whatever the outcome, and anything left over is swept at startup. Evidence crops
live next to the job database until the result expires or the user deletes it. All
directories are owner-only (0700), and files are named by analysis ID. The uploaded file's
name is never written anywhere.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

from starlette.datastructures import UploadFile

from deepfake_core.pipeline import EvidenceImage

_EVIDENCE_ID = re.compile(r"^e\d{1,2}$")
_CHUNK = 1 << 20


class UploadTooLargeError(Exception):
    pass


class Storage:
    def __init__(self, var_dir: Path) -> None:
        self.root = var_dir
        self.uploads = var_dir / "uploads"
        self.evidence = var_dir / "evidence"
        self.db_path = var_dir / "analyses.sqlite3"
        for directory in (self.root, self.uploads, self.evidence):
            directory.mkdir(parents=True, exist_ok=True)
            directory.chmod(0o700)

    def upload_path(self, analysis_id: str, extension: str) -> Path:
        return self.uploads / f"{analysis_id}{extension}"

    async def save_upload(self, upload: UploadFile, dest: Path, max_bytes: int) -> int:
        """Stream an upload to `dest`, stopping as soon as it exceeds `max_bytes`."""
        size = 0
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as out:
            while chunk := await upload.read(_CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadTooLargeError
                out.write(chunk)
        return size

    @staticmethod
    def delete_upload(path: Path) -> None:
        path.unlink(missing_ok=True)

    def sweep_uploads(self) -> int:
        """Delete every upload (run at startup: no upload may outlive its analysis)."""
        count = 0
        for path in self.uploads.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)
                count += 1
        return count

    def write_evidence(self, analysis_id: str, images: Sequence[EvidenceImage]) -> None:
        if not images:
            return
        directory = self.evidence / analysis_id
        directory.mkdir(mode=0o700, exist_ok=True)
        for image in images:
            (directory / f"{image.evidence_id}.jpg").write_bytes(image.jpeg)

    def evidence_file(self, analysis_id: str, evidence_id: str) -> Path | None:
        if not _EVIDENCE_ID.match(evidence_id):
            return None
        path = self.evidence / analysis_id / f"{evidence_id}.jpg"
        return path if path.is_file() else None

    def delete_evidence(self, analysis_id: str) -> None:
        shutil.rmtree(self.evidence / analysis_id, ignore_errors=True)
