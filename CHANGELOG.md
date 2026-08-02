# Changelog

All notable changes to AstroFlatAnalyzer are documented here.

## [0.9.4] - 2026-08-02

Enhancement release for responsive preview and full-resolution export
generation.

### Added

- Bounded multithreaded Gaussian filtering for preview and full-resolution
  illumination maps.
- Background Agg rendering, encoding, and atomic generation-safe export saves.
- Full-export benchmarks and regression coverage for numerical and raster
  fidelity.

### Changed

- Export contour lines are twice as wide, with contour labels approximately
  33% larger.
- Export resampling reuses the existing interpolation path with preallocated
  output storage.

## [0.9.3] - 2026-08-02

Enhancement release for responsive controls and preview rendering.

### Added

- Synchronized numeric inputs for the smoothing, contour-step, and JPG-quality
  sliders.
- Debounced preview recalculation while sliders are moving, with immediate
  refresh on release.
- Integer-factor, anti-aliased preview reduction while preserving
  full-resolution exports and corner-region analysis.

## [0.9.2] - 2026-08-01

Patch release adding application branding.

### Added

- AstroFlatAnalyzer application icon for macOS, Windows, and Linux packaging.
- Generated `.icns` and `.ico` platform assets from the supplied 1024×1024 PNG.

## [0.9.1] - 2026-08-01

Patch release for standalone drag-and-drop support.

### Fixed

- Enabled cross-platform drag-and-drop using the bundled TkDnD extension.
- Added `tkinterdnd2` to source dependencies and PyInstaller data collection so
  developers and standalone users do not need to install it separately.

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
