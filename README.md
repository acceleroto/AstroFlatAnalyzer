# AstroFlatAnalyzer

AstroFlatAnalyzer analyzes illumination falloff and vignetting in astronomy flat calibration frames. It is a standalone desktop application for macOS, Windows, and Linux that produces a smoothed grayscale illumination map with optional topographic-style contour lines.

FITS files are processed locally. The application does not upload images or require an internet connection after installation.

Current release: **0.9.2 pre-release**

- Repository: <https://github.com/Acceleroto/AstroFlatAnalyzer>
- Downloads: <https://github.com/Acceleroto/AstroFlatAnalyzer/releases>

## Features

- Load mono, Bayer, or already-debayered RGB FITS files (`.fits`, `.fit`, `.fts`)
- Read scaled FITS data, including files using `BZERO` and `BSCALE`
- Detect common Bayer patterns: `RGGB`, `BGGR`, `GRBG`, and `GBRG`
- Average each Bayer 2×2 cell into one analysis pixel before calculating illumination
- Normalize the map against a robust central reference region, treated as 100%
- Gaussian smoothing on the raw analysis image
- Optional contour lines at configurable percentage intervals
- Secondary 25×25-region illumination summary for all four corners
- Export PNG or JPG images
- User-selected export width with height automatically calculated to preserve the source aspect ratio
- Native file picker and TkDnD-based drag-and-drop FITS loading
- Branded application icons for packaged desktop builds

## Analysis behavior

### Bayer frames

For a 2D FITS file with a recognized `BAYERPAT` header, each aligned 2×2 pixel block is averaged:

```text
R G        1 analysis pixel
G B   →    (R + G + G + B) / 4
```

This reduces color-dependent Bayer variation before calculating the illumination profile. If the source dimensions are odd, the final row or column is omitted from the Bayer reduction.

Already-debayered 3D RGB FITS data is converted to luminance using BT.709 coefficients. Ordinary 2D mono FITS data is analyzed without Bayer reduction.

### Illumination map

The raw analysis image is smoothed first, then normalized. The median of a small central region is used as the reference, so the center is approximately 100% without relying on one potentially defective pixel.

The default controls are:

- Smoothing: **100 px**, adjustable from 25–200 px
- Display range: **80–100%**
- Contours: enabled
- Contour step: **2%**
- Contour minimum: internally set to 0%; levels outside the actual data range are omitted

For Bayer-reduced files, the smoothing value is expressed in original sensor pixels and converted internally to analysis-image pixels.

### Corner illumination summary

The GUI also reports the mean illumination of:

- The central 25×25 analysis-pixel region, treated as 100%
- The 25×25 regions in the Top Left, Top Right, Bottom Left, and Bottom Right corners

This summary uses the Bayer-reduced image, when applicable, before display smoothing. On a Bayer frame, a 25×25 analysis region corresponds to approximately 50×50 original sensor pixels.

## Running from source

Python 3.11 or newer is required.

### macOS with Homebrew Python

Homebrew Python does not include Tkinter unless the matching Tk package is installed:

```bash
brew install python-tk@3.13
```

Create the environment using the Tkinter-enabled Homebrew Python:

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python main.py
```

If you see `ModuleNotFoundError: No module named '_tkinter'`, install `python-tk@3.13` and recreate `.venv`.

### Windows

Use PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python main.py
```

### Linux

Install Tkinter through the distribution package manager first. On Ubuntu/Debian:

```bash
sudo apt install python3-tk
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python main.py
```

### Using the application

1. Click **Open FITS…**, or drag a FITS file onto the application window.
2. Adjust smoothing, display range, contour visibility, and contour step.
3. Choose an export resolution preset or enter an output width.
4. Select PNG or JPG.
5. Click **Save image…**.

Drag-and-drop is provided by the bundled TkDnD extension and is configured
automatically for source and standalone builds. Processing occurs after a
slider is released. Text fields update after pressing Enter or leaving the
field.

## Tests

Run the unit test suite from the project directory:

```bash
pytest -q
```

The tests cover mono FITS, RGB FITS, Bayer reduction, FITS scaling keywords, illumination normalization, smoothing, contour levels, export dimensions, and corner falloff calculations.

## Standalone builds

Standalone builds bundle Python, the application dependencies, and the native
TkDnD libraries. Users do not need to install Python, NumPy, SciPy, Astropy,
CustomTkinter, or `tkinterdnd2`.

PyInstaller builds must run on the target operating system and CPU architecture. The v0.9.2 release artifacts are:

```text
AstroFlatAnalyzer-macOS-AppleSilicon.zip
AstroFlatAnalyzer-macOS-Intel.zip
AstroFlatAnalyzer-Windows.zip
AstroFlatAnalyzer-Linux.tar.gz
```

### Local macOS build

On an Apple Silicon Mac:

```bash
source .venv/bin/activate
python -m PyInstaller --noconfirm flat_analyzer.spec
open dist/AstroFlatAnalyzer.app
```

The Apple Silicon build is produced by the native Apple Silicon Python environment. The Intel build should be produced by the Intel GitHub Actions runner or an Intel Mac.

### Running a downloaded build

#### macOS

1. Unzip the downloaded archive.
2. Move `AstroFlatAnalyzer.app` to Applications if desired.
3. Double-click it.

Unsigned builds may be blocked by Gatekeeper on first launch. Right-click the app, choose **Open**, and confirm.

#### Windows

1. Unzip the complete `AstroFlatAnalyzer-Windows` folder.
2. Keep the folder contents together.
3. Double-click `AstroFlatAnalyzer.exe`.

Windows SmartScreen may display a warning for unsigned builds. Choose **More info**, then **Run anyway**.

#### Linux

Extract the archive, keep the application directory intact, and run the bundled executable:

```bash
tar -xzf AstroFlatAnalyzer-Linux.tar.gz
cd AstroFlatAnalyzer
chmod +x AstroFlatAnalyzer
./AstroFlatAnalyzer
```

The Linux build requires a graphical desktop session and its normal system graphics libraries. An AppImage may be provided later for easier Linux distribution.

## GitHub Actions release builds

Cross-platform builds run in GitHub Actions using these hosted runners:

| Target | Runner |
|---|---|
| macOS Apple Silicon | `macos-15` |
| macOS Intel | `macos-15-intel` |
| Windows 64-bit | `windows-2025` |
| Linux x86_64 | `ubuntu-22.04` |

The workflow runs tests and a wheel smoke test before building. A manual workflow run produces downloadable test artifacts. Pushing a `v0.9.2` tag produces the four archives, a `SHA256SUMS.txt` file, and a GitHub pre-release. macOS Intel support is retained as a separate build because Apple Silicon and Intel Python dependencies cannot safely be mixed.

## License

MIT
