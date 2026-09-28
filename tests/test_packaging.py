"""CI tests the dependency versions Waveguide Generator ships."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def _name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0].strip().lower()


def test_every_runtime_dependency_is_pinned_for_the_wg_ci_job():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = {
        _name(requirement)
        for requirement in project["project"]["dependencies"]
        if "python_version" not in requirement  # tomli: not on WG's 3.13
    }
    pins = {}
    for line in (ROOT / ".github" / "constraints-wg.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, version = line.partition("==")
            assert version, f"constraint is not an exact pin: {line!r}"
            pins[name.strip().lower()] = version.strip()

    assert set(pins) == runtime


def test_ci_installs_the_wg_constraints():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "-c .github/constraints-wg.txt" in workflow
