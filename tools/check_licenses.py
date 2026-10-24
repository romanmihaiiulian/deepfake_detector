"""Check dependencies/licenses.yml against what is actually installed.

Run it inside the workspace environment: `uv run python tools/check_licenses.py`.

Python packages are read from importlib.metadata of the running interpreter; npm packages
from apps/web/package-lock.json. Hand-curated entries (models, datasets, tools, system
libraries, research references) are only validated for shape.

`distributed` is computed, not typed in: a Python package is distributed when it is in the
runtime dependency closure of deepfake-core or deepfake-api, and an npm package when it is
a production (non-dev) dependency of the web app, i.e. it ends up in the built bundle.

The check fails when:
  * an installed package is missing from the inventory, or its version has drifted;
  * an inventory package entry is no longer installed;
  * an entry's `distributed` flag disagrees with the computed value;
  * a distributed entry is not cleared for commercial use, or its license is copyleft,
    non-commercial, research-only or unknown and its notes carry no "EXCEPTION:" sign-off.

`--update` rewrites package entries to match the environment: new packages are added as
`review_required`, and an entry whose detected license changed goes back to review.
Curated entries are never modified. `--approve-permissive` marks package entries whose
license consists only of well-known permissive SPDX identifiers as reviewed today.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "dependencies" / "licenses.yml"
NPM_LOCK = REPO / "apps" / "web" / "package-lock.json"

FIELDS = (
    "name",
    "kind",
    "version",
    "source",
    "license",
    "commercial_use",
    "distributed",
    "review_date",
    "notes",
)
PACKAGE_KINDS = ("python", "npm")
CURATED_KINDS = ("model", "dataset", "tool", "system", "reference")
COMMERCIAL_USE = ("yes", "no", "review_required")

WORKSPACE_PACKAGES = frozenset({"deepfake-core", "deepfake-api", "deepfake-bench"})
RUNTIME_ROOTS = ("deepfake-core", "deepfake-api")

DENYLIST = re.compile(
    r"(\bA?GPL|\bLGPL|\bSSPL|\bEUPL|CC-BY-NC|NonCommercial|Non-Commercial|research|"
    r"\bUNKNOWN\b|\bproprietary\b|OpenRAIL|ResearchRAIL)",
    re.IGNORECASE,
)
PERMISSIVE = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "Apache-2.0 WITH LLVM-exception",
        "BlueOak-1.0.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "BSL-1.0",
        "CC0-1.0",
        "CNRI-Python",
        "HPND",
        "ISC",
        "MIT",
        "MIT-0",
        "MIT-CMU",
        "PSF-2.0",
        "Python-2.0",
        "Unlicense",
        "Zlib",
    }
)

CLASSIFIER_SPDX = {
    "Apache Software License": "Apache-2.0",
    "BSD License": "BSD",
    "ISC License (ISCL)": "ISC",
    "MIT License": "MIT",
    "MIT No Attribution License (MIT-0)": "MIT-0",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "Python Software Foundation License": "PSF-2.0",
    "The Unlicense (Unlicense)": "Unlicense",
    "zlib/libpng License": "Zlib",
}
LICENSE_FIELD_SPDX = {
    "apache 2.0": "Apache-2.0",
    "apache 2.0 license": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "bsd": "BSD",
    "isc license": "ISC",
    "mit": "MIT",
    "mit license": "MIT",
}

HEADER = """\
# Machine-readable license inventory (see docs/licensing.md).
#
# Every dependency, model weight, dataset and tool used by the project is listed here.
# Package entries (kind: python | npm) are maintained by `tools/check_licenses.py --update`;
# curated entries (kind: model | dataset | tool | system | reference) are written by hand.
#
#   commercial_use: yes | no | review_required
#   distributed:    true when it ships with the product (runtime closure of deepfake-core /
#                   deepfake-api, or the web bundle); computed for packages.
#   notes:          an "EXCEPTION:" note is a signed-off decision to ship a license the
#                   checker would otherwise reject.
"""


@dataclass(frozen=True, slots=True)
class Installed:
    name: str
    kind: str
    version: str
    license: str
    source: str
    distributed: bool


@dataclass(slots=True)
class Report:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------- discovery


def detect_license(metadata: Any) -> str:
    """Best-effort SPDX expression from core metadata (PEP 639 first, then legacy fields)."""
    expression = metadata.get("License-Expression")
    if expression:
        return str(expression).strip()
    field = (metadata.get("License") or "").strip()
    if field and "\n" not in field and len(field) <= 64:
        mapped = LICENSE_FIELD_SPDX.get(field.lower())
        if mapped:
            return mapped
        if field.upper() != "UNKNOWN" and field.lower() != "dual license":
            return field
    licenses = []
    for classifier in metadata.get_all("Classifier") or []:
        if classifier.startswith("License ::"):
            label = classifier.split("::")[-1].strip()
            if label != "OSI Approved":
                licenses.append(CLASSIFIER_SPDX.get(label, label))
    if licenses:
        return " OR ".join(sorted(set(licenses)))
    return "UNKNOWN"


def runtime_closure(roots: Iterable[str]) -> set[str]:
    """Canonical names of every installed distribution reachable from `roots`.

    Environment markers are evaluated for this interpreter, and extras are followed only
    when a parent requests them.
    """
    environment = {key: str(value) for key, value in default_environment().items()}
    seen: set[tuple[str, frozenset[str]]] = set()
    names: set[str] = set()
    stack = [(canonicalize_name(root), frozenset[str]()) for root in roots]
    while stack:
        name, extras = stack.pop()
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        names.add(name)
        for spec in dist.requires or []:
            try:
                requirement = Requirement(spec)
            except InvalidRequirement:
                continue
            if requirement.marker is not None and not any(
                requirement.marker.evaluate({**environment, "extra": extra})
                for extra in (extras or {""})
            ):
                continue
            stack.append((canonicalize_name(requirement.name), frozenset(requirement.extras)))
    return names - WORKSPACE_PACKAGES


def installed_python() -> dict[tuple[str, str], Installed]:
    distributed = runtime_closure(RUNTIME_ROOTS)
    found: dict[tuple[str, str], Installed] = {}
    for dist in md.distributions():
        name = canonicalize_name(dist.metadata["Name"])
        if name in WORKSPACE_PACKAGES:
            continue
        found[("python", name)] = Installed(
            name=name,
            kind="python",
            version=dist.version,
            license=detect_license(dist.metadata),
            source=f"https://pypi.org/project/{name}/",
            distributed=name in distributed,
        )
    return found


def installed_npm(lock_path: Path = NPM_LOCK) -> dict[tuple[str, str], Installed]:
    """npm packages from package-lock.json (lockfile v2/v3 `packages` map)."""
    if not lock_path.is_file():
        return {}
    lock = json.loads(lock_path.read_text())
    versions: dict[str, set[str]] = {}
    licenses: dict[str, set[str]] = {}
    production: dict[str, bool] = {}
    for key, meta in lock.get("packages", {}).items():
        if not key or "node_modules/" not in key:
            continue  # the root project itself, or a workspace link
        name = key.rsplit("node_modules/", 1)[-1]
        versions.setdefault(name, set()).add(str(meta.get("version", "")))
        raw = meta.get("license", "UNKNOWN")
        licenses.setdefault(name, set()).add(raw if isinstance(raw, str) else json.dumps(raw))
        is_prod = not (meta.get("dev") or meta.get("devOptional"))
        production[name] = production.get(name, False) or is_prod
    return {
        ("npm", name): Installed(
            name=name,
            kind="npm",
            version=", ".join(sorted(versions[name])),
            license=" OR ".join(sorted(licenses[name])),
            source=f"https://www.npmjs.com/package/{name}",
            distributed=production[name],
        )
        for name in versions
    }


# ---------------------------------------------------------------------------- inventory


def load_inventory(path: Path = INVENTORY) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    entries: list[dict[str, Any]] = list(data.get("entries") or [])
    for entry in entries:
        if isinstance(entry.get("review_date"), dt.date):
            entry["review_date"] = entry["review_date"].isoformat()
        if isinstance(entry.get("commercial_use"), bool):  # YAML 1.1 reads bare yes/no as bool
            entry["commercial_use"] = "yes" if entry["commercial_use"] else "no"
        if entry.get("version") is not None:
            entry["version"] = str(entry["version"])
    return entries


def dump_inventory(entries: list[dict[str, Any]], path: Path = INVENTORY) -> None:
    order = {kind: i for i, kind in enumerate(CURATED_KINDS + PACKAGE_KINDS)}
    rows = []
    for entry in sorted(entries, key=lambda e: (order.get(e["kind"], 99), e["name"].lower())):
        row = {field: entry.get(field) for field in FIELDS}
        if isinstance(row["review_date"], str):
            row["review_date"] = dt.date.fromisoformat(row["review_date"])
        rows.append(row)
    body = yaml.safe_dump(
        {"schema_version": 1, "entries": rows},
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    path.write_text(HEADER + "\n" + body)


def key_of(entry: Mapping[str, Any]) -> tuple[str, str]:
    name = str(entry["name"])
    if entry["kind"] == "python":
        name = canonicalize_name(name)
    return (str(entry["kind"]), name)


# ---------------------------------------------------------------------------- checking


def check(
    entries: list[dict[str, Any]],
    installed: Mapping[tuple[str, str], Installed],
    *,
    kinds: Iterable[str] = PACKAGE_KINDS,
) -> Report:
    """Compare the inventory with the installed packages of the given kinds."""
    kinds = tuple(kinds)
    report = Report(errors=[], warnings=[])
    by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for entry in entries:
        label = f"{entry.get('kind')}:{entry.get('name')}"
        missing = [field for field in FIELDS if field not in entry]
        if missing:
            report.errors.append(f"{label}: missing fields {missing}")
            continue
        if entry["kind"] not in PACKAGE_KINDS + CURATED_KINDS:
            report.errors.append(f"{label}: unknown kind {entry['kind']!r}")
            continue
        if entry["commercial_use"] not in COMMERCIAL_USE:
            report.errors.append(f"{label}: commercial_use must be one of {COMMERCIAL_USE}")
        if not isinstance(entry["distributed"], bool):
            report.errors.append(f"{label}: distributed must be true or false")
        key = key_of(entry)
        if key in by_key:
            report.errors.append(f"{label}: duplicate entry")
        by_key[key] = entry

        if entry["distributed"] is True:
            if entry["commercial_use"] != "yes":
                report.errors.append(
                    f"{label}: distributed but commercial_use is {entry['commercial_use']!r}"
                )
            license_text = str(entry["license"] or "UNKNOWN")
            if DENYLIST.search(license_text) and "EXCEPTION:" not in str(entry["notes"] or ""):
                report.errors.append(
                    f"{label}: distributed with license {license_text!r} and no EXCEPTION note"
                )
        elif entry["commercial_use"] == "review_required":
            report.warnings.append(f"{label}: review_required (not distributed)")

    for key, package in sorted(installed.items()):
        if key[0] not in kinds:
            continue
        inventoried = by_key.get(key)
        label = f"{key[0]}:{key[1]}"
        if inventoried is None:
            report.errors.append(f"{label} {package.version} is installed but not inventoried")
            continue
        if str(inventoried["version"]) != package.version:
            report.errors.append(
                f"{label}: inventory has version {inventoried['version']}, "
                f"installed {package.version}"
            )
        if inventoried["distributed"] is not package.distributed:
            report.errors.append(
                f"{label}: inventory says distributed={inventoried['distributed']}, "
                f"computed {package.distributed}"
            )

    for key in sorted(by_key):
        if key[0] in kinds and key not in installed:
            report.errors.append(f"{key[0]}:{key[1]} is inventoried but not installed")
    return report


def update(
    entries: list[dict[str, Any]],
    installed: Mapping[tuple[str, str], Installed],
    *,
    kinds: Iterable[str] = PACKAGE_KINDS,
) -> list[dict[str, Any]]:
    """Return a new entry list whose package entries match `installed`."""
    kinds = tuple(kinds)
    existing = {key_of(entry): entry for entry in entries}
    result = [dict(entry) for entry in entries if entry["kind"] not in kinds]
    for key, package in installed.items():
        if key[0] not in kinds:
            continue
        previous = existing.get(key)
        if previous is None:
            result.append(
                {
                    "name": package.name,
                    "kind": package.kind,
                    "version": package.version,
                    "source": package.source,
                    "license": package.license,
                    "commercial_use": "review_required",
                    "distributed": package.distributed,
                    "review_date": None,
                    "notes": "",
                }
            )
            continue
        entry = dict(previous)
        if str(entry["version"]) != package.version:
            if package.license != entry["license"]:
                entry["license"] = package.license
                entry["commercial_use"] = "review_required"
                entry["review_date"] = None
            entry["version"] = package.version
        entry["distributed"] = package.distributed
        result.append(entry)
    return result


def is_permissive(expression: str) -> bool:
    terms = re.split(r"\s+(?:AND|OR)\s+", expression.replace("(", " ").replace(")", " ").strip())
    return bool(terms) and all(term.strip() in PERMISSIVE for term in terms)


def approve_permissive(entries: list[dict[str, Any]], today: dt.date) -> int:
    approved = 0
    for entry in entries:
        if (
            entry["kind"] in PACKAGE_KINDS
            and entry["commercial_use"] == "review_required"
            and is_permissive(str(entry["license"]))
        ):
            entry["commercial_use"] = "yes"
            entry["review_date"] = today.isoformat()
            entry["notes"] = entry.get("notes") or "permissive license per package metadata"
            approved += 1
    return approved


# ---------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true", help="sync package entries")
    parser.add_argument(
        "--approve-permissive",
        action="store_true",
        help="mark package entries with permissive licenses as reviewed today",
    )
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--npm-lock", type=Path, default=NPM_LOCK)
    args = parser.parse_args(argv)

    entries = load_inventory(args.inventory)
    installed = {**installed_python(), **installed_npm(args.npm_lock)}
    kinds = ("python", "npm") if args.npm_lock.is_file() else ("python",)

    if args.update or args.approve_permissive:
        if args.update:
            entries = update(entries, installed, kinds=kinds)
        if args.approve_permissive:
            count = approve_permissive(entries, dt.date.today())
            print(f"approved {count} permissive package entries")
        dump_inventory(entries, args.inventory)
        entries = load_inventory(args.inventory)

    report = check(entries, installed, kinds=kinds)
    for warning in report.warnings:
        print(f"warning: {warning}")
    for error in report.errors:
        print(f"error: {error}", file=sys.stderr)
    distributed = sum(1 for entry in entries if entry.get("distributed") is True)
    print(
        f"{len(entries)} inventory entries ({distributed} distributed): "
        f"{len(report.errors)} errors, {len(report.warnings)} warnings"
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
