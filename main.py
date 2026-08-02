"""AstroFlatAnalyzer entry point."""

from __future__ import annotations

import sys


def self_test() -> int:
    """Validate bundled runtime dependencies without opening a window."""
    import astropy
    import customtkinter
    import matplotlib
    import numpy
    import scipy
    import tkinterdnd2

    from flat_analyzer import __version__

    dependencies = (astropy, customtkinter, matplotlib, numpy, scipy, tkinterdnd2)
    if any(module is None for module in dependencies):
        raise RuntimeError("A bundled dependency could not be imported.")

    print(f"AstroFlatAnalyzer {__version__} runtime self-test passed")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()

    from flat_analyzer.ui.main_window import run_app

    run_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
