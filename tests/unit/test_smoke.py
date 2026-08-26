"""Smoke tests: package structure and core dependency health.

These tests prove the install is coherent and the package is importable.
They require no GPU, no network, no datasets, and no robot.
"""

import importlib
import re


def test_package_importable() -> None:
    mod = importlib.import_module("langgraph_vla_agent")
    assert mod is not None


def test_version_present_and_semver() -> None:
    from langgraph_vla_agent import __version__

    assert isinstance(__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", __version__), (
        f"__version__ {__version__!r} does not look like semver"
    )


def test_pydantic_available_and_validates() -> None:
    """Core dep for typed domain models is importable and functional."""
    import pytest
    from pydantic import BaseModel, ValidationError

    class _Probe(BaseModel):
        value: int
        label: str = "default"

    p = _Probe(value=42)
    assert p.value == 42
    assert p.label == "default"

    with pytest.raises(ValidationError):
        _Probe(value="not-an-int")  # type: ignore[arg-type]


def test_structlog_available() -> None:
    """Structured logging dep is importable."""
    import structlog

    logger = structlog.get_logger("smoke")
    # Calling bind() exercises the lazy-proxy; it must not raise.
    bound = logger.bind(component="smoke_test")
    assert bound is not None


def test_numpy_available() -> None:
    """NumPy dep is importable and the required version is present."""
    import numpy as np

    arr = np.zeros((3, 4), dtype=np.float32)
    assert arr.shape == (3, 4)

    # LeRobot requires numpy>=2.0; verify we're on a compatible version.
    major = int(np.__version__.split(".")[0])
    assert major >= 2, f"numpy>= 2.0 required, got {np.__version__}"
