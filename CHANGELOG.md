# Changelog

All notable changes to AstroFlatAnalyzer are documented here.

## [0.9.0] - 2026-08-01

Pre-release for standalone-build testing.

### Added

- CustomTkinter desktop application for macOS, Windows, and Linux.
- Mono, Bayer, and already-debayered RGB FITS loading.
- FITS `BZERO`/`BSCALE` handling.
- Bayer-aware 2×2 block averaging for recognized CFA patterns.
- Center-normalized illumination maps with configurable Gaussian smoothing.
- Optional percentage contour overlays.
- 25×25-region corner illumination summary.
- PNG and JPG export with aspect-ratio-locked output sizing.
- PyInstaller packaging configuration and cross-platform GitHub Actions builds.

### Known distribution limitations

- Builds are unsigned; macOS Gatekeeper and Windows SmartScreen may display warnings.
- Linux is distributed as a portable tarball rather than an AppImage.
- Drag-and-drop support depends on the platform's Tk integration; the Open FITS
  file picker is the guaranteed input method.
- Standalone artifacts must be built for the matching operating system and CPU
  architecture.

## [1.0.0]

Reserved for the first release after standalone artifacts have been tested on
all supported platforms.
