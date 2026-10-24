"""Exception types raised by the engine."""

from __future__ import annotations


class DeepfakeCoreError(Exception):
    """Base class for engine errors."""


class VideoValidationError(DeepfakeCoreError):
    """The input is not a video the engine can analyze.

    `code` is stable and machine-readable (the API returns it to clients); `message` is
    human-readable and never contains the uploaded file's name.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ModelMissingError(DeepfakeCoreError):
    """A required model file is not present in the models directory."""


class ModelIntegrityError(DeepfakeCoreError):
    """A model file does not match its pinned size or checksum."""
