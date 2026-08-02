"""Release metadata and bundled-runtime checks."""

from pathlib import Path

from flat_analyzer import __version__
from main import self_test


def test_release_version():
    assert __version__ == "0.9.1"


def test_license_exists():
    assert (Path(__file__).parents[1] / "LICENSE").is_file()


def test_runtime_self_test():
    assert self_test() == 0
