"""FITS flat frame loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

FITS_EXTENSIONS = {".fits", ".fit", ".fts"}
SUPPORTED_BAYER_PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}

# ITU-R BT.709 luminance coefficients
_LUM_R, _LUM_G, _LUM_B = 0.2126, 0.7152, 0.0722


@dataclass(frozen=True)
class FlatMetadata:
    """Metadata extracted from a loaded flat frame."""

    path: Path
    nx: int
    ny: int
    bitpix: int | None
    is_rgb: bool
    bayerpat: str | None
    bayer_reduced: bool
    analysis_nx: int
    analysis_ny: int


@dataclass
class LoadedFlat:
    """A validated mono luminance image ready for analysis."""

    image: np.ndarray
    metadata: FlatMetadata


class FitsLoadError(Exception):
    """Raised when a FITS file cannot be loaded or validated."""


def _rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array (..., 3) to luminance."""
    return _LUM_R * rgb[..., 0] + _LUM_G * rgb[..., 1] + _LUM_B * rgb[..., 2]


def _extract_image_data(hdu: fits.PrimaryHDU | fits.ImageHDU) -> np.ndarray:
    data = np.asarray(hdu.data, dtype=np.float64)
    if data.ndim == 0:
        raise FitsLoadError("FITS HDU contains a scalar, not an image.")
    return data


def _reduce_bayer_2x2(data: np.ndarray) -> np.ndarray:
    """Average each aligned 2x2 Bayer cell into one illumination sample."""
    ny, nx = data.shape
    even_ny = ny - (ny % 2)
    even_nx = nx - (nx % 2)
    cropped = data[:even_ny, :even_nx]
    return cropped.reshape(even_ny // 2, 2, even_nx // 2, 2).mean(axis=(1, 3))


def _to_luminance_2d(
    data: np.ndarray,
    bayerpat: str | None,
) -> tuple[np.ndarray, bool, bool]:
    """Return a 2D illumination array plus RGB/Bayer processing flags."""
    if data.ndim == 2:
        normalized_pattern = (bayerpat or "").strip().upper()
        if normalized_pattern in SUPPORTED_BAYER_PATTERNS:
            return _reduce_bayer_2x2(data), False, True
        return data, False, False

    if data.ndim != 3:
        raise FitsLoadError(
            f"Expected 2D (mono) or 3D (RGB) image data, got {data.ndim} dimensions."
        )

    if data.shape[0] == 3:
        # (3, ny, nx) — channel-first
        luminance = _rgb_to_luminance(np.moveaxis(data, 0, -1))
        return luminance, True, False

    if data.shape[-1] == 3:
        # (ny, nx, 3) — channel-last
        luminance = _rgb_to_luminance(data)
        return luminance, True, False

    raise FitsLoadError(
        f"3D FITS data must have 3 channels on axis 0 or axis -1; shape is {data.shape}."
    )


def load_flat(path: str | Path) -> LoadedFlat:
    """Load a mono or RGB flat FITS file and return a 2D float luminance image."""
    file_path = Path(path).expanduser().resolve()

    if file_path.suffix.lower() not in FITS_EXTENSIONS:
        raise FitsLoadError(
            f"Unsupported file type '{file_path.suffix}'. "
            f"Expected one of: {', '.join(sorted(FITS_EXTENSIONS))}"
        )

    if not file_path.is_file():
        raise FitsLoadError(f"File not found: {file_path}")

    with fits.open(file_path, memmap=False) as hdul:
        if len(hdul) == 0:
            raise FitsLoadError("FITS file contains no HDUs.")

        hdu = hdul[0]
        if hdu.data is None:
            raise FitsLoadError("Primary HDU has no image data.")

        header = hdu.header
        bitpix = header.get("BITPIX")
        raw_bayerpat = header.get("BAYERPAT")
        bayerpat = str(raw_bayerpat).strip().upper() if raw_bayerpat is not None else None

        raw = _extract_image_data(hdu)
        image, is_rgb, bayer_reduced = _to_luminance_2d(raw, bayerpat)
        if bayer_reduced:
            source_ny, source_nx = raw.shape
        else:
            source_ny, source_nx = image.shape

        if not np.isfinite(image).any():
            raise FitsLoadError("Image contains no finite pixel values.")

        ny, nx = image.shape
        if ny < 2 or nx < 2:
            raise FitsLoadError(f"Image too small for analysis: {nx}×{ny} pixels.")

        metadata = FlatMetadata(
            path=file_path,
            nx=source_nx,
            ny=source_ny,
            bitpix=int(bitpix) if bitpix is not None else None,
            is_rgb=is_rgb,
            bayerpat=bayerpat,
            bayer_reduced=bayer_reduced,
            analysis_nx=nx,
            analysis_ny=ny,
        )

    return LoadedFlat(image=image, metadata=metadata)
