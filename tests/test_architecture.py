"""Module boundaries: the engine must stay independent of the API, the web layer and the bench."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

BOUNDARIES = {
    "packages/detector/src/deepfake_core": {
        "deepfake_api",
        "deepfake_bench",
        "fastapi",
        "starlette",
        "uvicorn",
        "sklearn",
        "matplotlib",
    },
    "apps/api/src/deepfake_api": {"deepfake_bench", "sklearn", "matplotlib"},
}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("package", sorted(BOUNDARIES))
def test_package_respects_import_boundary(package: str) -> None:
    forbidden = BOUNDARIES[package]
    violations = {
        str(path.relative_to(REPO)): sorted(imported_modules(path) & forbidden)
        for path in sorted((REPO / package).rglob("*.py"))
    }
    assert {file: mods for file, mods in violations.items() if mods} == {}
