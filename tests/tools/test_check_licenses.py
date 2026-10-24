from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from tools import check_licenses as cl


def entry(name: str, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": name,
        "kind": "python",
        "version": "1.0",
        "source": f"https://pypi.org/project/{name}/",
        "license": "MIT",
        "commercial_use": "yes",
        "distributed": True,
        "review_date": "2026-10-24",
        "notes": "",
    }
    base.update(overrides)
    return base


def installed(name: str, version: str = "1.0", lic: str = "MIT", distributed: bool = True) -> Any:
    return cl.Installed(name, "python", version, lic, "src", distributed)


def test_clean_inventory_passes() -> None:
    report = cl.check([entry("a")], {("python", "a"): installed("a")})
    assert report.ok, report.errors


def test_missing_and_stale_packages_fail() -> None:
    report = cl.check([entry("gone")], {("python", "new"): installed("new")})
    assert any("new 1.0 is installed but not inventoried" in e for e in report.errors)
    assert any("gone is inventoried but not installed" in e for e in report.errors)


def test_version_drift_fails() -> None:
    report = cl.check([entry("a", version="1.0")], {("python", "a"): installed("a", "2.0")})
    assert any("inventory has version 1.0, installed 2.0" in e for e in report.errors)


def test_distributed_flag_must_match_computed_closure() -> None:
    report = cl.check([entry("a")], {("python", "a"): installed("a", distributed=False)})
    assert any("computed False" in e for e in report.errors)


@pytest.mark.parametrize(
    "license_text",
    ["GPL-3.0-only", "AGPL-3.0", "LGPL-2.1-or-later", "CC-BY-NC-4.0", "UNKNOWN", "research-only"],
)
def test_distributed_copyleft_or_unknown_needs_exception(license_text: str) -> None:
    rejected = cl.check([entry("a", license=license_text)], {("python", "a"): installed("a")})
    assert not rejected.ok
    excepted = cl.check(
        [entry("a", license=license_text, notes="EXCEPTION: dynamically linked, approved")],
        {("python", "a"): installed("a")},
    )
    assert excepted.ok, excepted.errors


def test_distributed_entry_must_be_cleared_for_commercial_use() -> None:
    report = cl.check(
        [entry("a", commercial_use="review_required")], {("python", "a"): installed("a")}
    )
    assert any("commercial_use is 'review_required'" in e for e in report.errors)


def test_curated_entries_are_shape_checked_but_not_matched_to_packages() -> None:
    model = entry("Some weights", kind="model", distributed=False, commercial_use="no")
    report = cl.check([model], {})
    assert report.ok
    broken = cl.check([{"name": "x", "kind": "model"}], {})
    assert any("missing fields" in e for e in broken.errors)


def test_update_adds_new_packages_for_review_and_keeps_curated_entries() -> None:
    curated = entry("weights", kind="model", distributed=False)
    updated = cl.update([curated, entry("old")], {("python", "new"): installed("new", lic="BSD")})
    by_name = {e["name"]: e for e in updated}
    assert by_name["weights"] == curated
    assert "old" not in by_name
    assert by_name["new"]["commercial_use"] == "review_required"
    assert by_name["new"]["license"] == "BSD"


def test_update_resets_review_only_when_license_changes() -> None:
    same = cl.update([entry("a")], {("python", "a"): installed("a", "2.0")})[0]
    assert (same["version"], same["commercial_use"]) == ("2.0", "yes")
    changed = cl.update([entry("a")], {("python", "a"): installed("a", "2.0", lic="GPL-3.0")})[0]
    assert (changed["license"], changed["commercial_use"]) == ("GPL-3.0", "review_required")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("MIT", True),
        ("BSD-3-Clause AND 0BSD AND MIT", True),
        ("Apache-2.0 OR BSD-2-Clause", True),
        ("MPL-2.0", False),
        ("BSD", False),
        ("MIT AND GPL-2.0-only", False),
    ],
)
def test_is_permissive(expression: str, expected: bool) -> None:
    assert cl.is_permissive(expression) is expected


def test_approve_permissive_only_touches_permissive_package_entries() -> None:
    entries = [
        entry("a", commercial_use="review_required", review_date=None),
        entry("b", license="MPL-2.0", commercial_use="review_required", review_date=None),
    ]
    assert cl.approve_permissive(entries, dt.date(2026, 10, 24)) == 1
    assert entries[0]["commercial_use"] == "yes"
    assert entries[1]["commercial_use"] == "review_required"


def test_inventory_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "licenses.yml"
    original = [entry("a", commercial_use="no", review_date="2026-10-24")]
    cl.dump_inventory(original, path)
    text = path.read_text()
    assert text.startswith("# Machine-readable license inventory")
    assert "review_date: 2026-10-24" in text
    assert cl.load_inventory(path) == original


def test_npm_lockfile_parsing(tmp_path: Path) -> None:
    lock = {
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "deepfake-web"},
            "node_modules/react": {"version": "19.0.0", "license": "MIT"},
            "node_modules/vite": {"version": "7.0.0", "license": "MIT", "dev": True},
            "node_modules/a/node_modules/react": {"version": "18.0.0", "license": "MIT"},
        },
    }
    path = tmp_path / "package-lock.json"
    path.write_text(json.dumps(lock))
    found = cl.installed_npm(path)
    assert found[("npm", "react")].version == "18.0.0, 19.0.0"
    assert found[("npm", "react")].distributed is True
    assert found[("npm", "vite")].distributed is False


def test_detect_license_prefers_pep639_expression() -> None:
    class Meta(dict[str, Any]):
        def get_all(self, key: str) -> list[str]:
            value = self.get(key)
            return value if isinstance(value, list) else []

    assert cl.detect_license(Meta({"License-Expression": "Apache-2.0"})) == "Apache-2.0"
    assert cl.detect_license(Meta({"License": "Apache 2.0"})) == "Apache-2.0"
    classifiers = Meta({"Classifier": ["License :: OSI Approved :: MIT License"]})
    assert cl.detect_license(classifiers) == "MIT"
    assert cl.detect_license(Meta({})) == "UNKNOWN"


def test_repository_inventory_is_consistent_with_this_environment() -> None:
    """The real inventory must match the installed workspace (same check as `make licenses`)."""
    kinds = ("python", "npm") if cl.NPM_LOCK.is_file() else ("python",)
    installed_now = {**cl.installed_python(), **cl.installed_npm()}
    report = cl.check(cl.load_inventory(), installed_now, kinds=kinds)
    assert report.ok, "\n".join(report.errors)
