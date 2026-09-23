"""Guards that keep declared dependency majors honest.

An open-ended `>=` delegates the choice of major to upstream release timing. That is
exactly how `stereohand` moved to OpenCV 5 unannounced — every test stayed green and live
calibration crashed the moment a ChArUco board entered frame. These two tests make that
class of drift a red build instead of a surprise at the hardware.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import re
import tomllib
from pathlib import Path

import pytest

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _declared_requirements() -> dict[str, str]:
    """Every declared requirement (runtime + extras) as ``{distribution: specifier}``."""
    project = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]
    specifiers = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        specifiers.extend(extra)
    return {re.split(r"[><=!~@\[]", s, maxsplit=1)[0].strip(): s for s in specifiers}


def _version_pinned_requirements() -> dict[str, str]:
    """Only the requirements carrying a version specifier (i.e. excluding direct URLs)."""
    return {name: spec for name, spec in _declared_requirements().items() if "@" not in spec}


@pytest.mark.parametrize("distribution", sorted(_version_pinned_requirements()))
def test_every_dependency_declares_an_upper_bound(distribution: str) -> None:
    specifier = _version_pinned_requirements()[distribution]
    assert "<" in specifier, f"{specifier!r} has no upper bound — pin the tested major"


def test_direct_url_dependencies_are_pinned_to_a_tag() -> None:
    """A git dep without a tag re-resolves to whatever the default branch holds that day."""
    url_requirements = {
        name: spec for name, spec in _declared_requirements().items() if "@ git+" in spec
    }
    assert url_requirements, "expected at least the stereohand git dependency"
    for name, specifier in url_requirements.items():
        assert re.search(r"@v\d+\.\d+", specifier), f"{name} is not pinned to a version tag"


@pytest.mark.parametrize("distribution", sorted(_version_pinned_requirements()))
def test_installed_version_satisfies_the_declaration(distribution: str) -> None:
    packaging_requirements = pytest.importorskip("packaging.requirements")
    specifier = _version_pinned_requirements()[distribution]
    try:
        installed = importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        pytest.skip(f"{distribution} not installed (optional extra)")
    requirement = packaging_requirements.Requirement(specifier)
    assert requirement.specifier.contains(installed, prereleases=True), (
        f"pyproject declares {specifier!r} but {distribution} {installed} is installed"
    )
