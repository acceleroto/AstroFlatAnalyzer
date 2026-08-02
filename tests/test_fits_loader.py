"""Tests for FITS loading."""

import numpy as np
import pytest
from astropy.io import fits

from flat_analyzer.fits_loader import FitsLoadError, load_flat


def _write_fits(path, data: np.ndarray) -> None:
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    hdu.writeto(path, overwrite=True)


def test_load_mono_fits(tmp_path):
    data = np.random.uniform(500, 1500, (100, 150)).astype(np.float32)
    path = tmp_path / "mono.fits"
    _write_fits(path, data)

    loaded = load_flat(path)
    assert loaded.image.shape == (100, 150)
    assert not loaded.metadata.is_rgb
    assert loaded.metadata.nx == 150
    assert loaded.metadata.ny == 100
    assert not loaded.metadata.bayer_reduced


def test_load_rgb_channel_first(tmp_path):
    ny, nx = 80, 120
    rgb = np.random.uniform(0, 1, (3, ny, nx)).astype(np.float32)
    path = tmp_path / "rgb_cf.fits"
    _write_fits(path, rgb)

    loaded = load_flat(path)
    assert loaded.image.shape == (ny, nx)
    assert loaded.metadata.is_rgb


def test_load_rgb_channel_last(tmp_path):
    ny, nx = 80, 120
    rgb = np.random.uniform(0, 1, (ny, nx, 3)).astype(np.float32)
    path = tmp_path / "rgb_cl.fits"
    _write_fits(path, rgb)

    loaded = load_flat(path)
    assert loaded.image.shape == (ny, nx)
    assert loaded.metadata.is_rgb


def test_reject_4d_fits(tmp_path):
    data = np.zeros((2, 3, 10, 10), dtype=np.float32)
    path = tmp_path / "bad.fits"
    _write_fits(path, data)

    with pytest.raises(FitsLoadError):
        load_flat(path)


def test_file_not_found():
    with pytest.raises(FitsLoadError):
        load_flat("/nonexistent/path/file.fits")


def test_load_fits_with_bzero_bscale(tmp_path):
    """Camera flats often use integer pixels with BZERO/BSCALE keywords."""
    data = np.random.randint(1000, 60000, (100, 150), dtype=np.uint16)
    hdu = fits.PrimaryHDU(data)
    hdu.header["BZERO"] = 32768
    hdu.header["BSCALE"] = 1
    path = tmp_path / "scaled.fits"
    hdu.writeto(path, overwrite=True)

    loaded = load_flat(path)
    assert loaded.image.shape == (100, 150)
    assert loaded.image.dtype == np.float64
    assert loaded.image.min() >= 0


def test_load_bayer_fits_averages_each_2x2_cell(tmp_path):
    data = np.array(
        [
            [100, 200, 200, 400],
            [300, 400, 600, 800],
            [50, 100, 400, 800],
            [150, 200, 1200, 1600],
        ],
        dtype=np.float32,
    )
    hdu = fits.PrimaryHDU(data)
    hdu.header["BAYERPAT"] = "RGGB"
    path = tmp_path / "bayer.fits"
    hdu.writeto(path)

    loaded = load_flat(path)

    np.testing.assert_allclose(loaded.image, [[250, 500], [125, 1000]])
    assert loaded.metadata.bayer_reduced
    assert loaded.metadata.bayerpat == "RGGB"
    assert (loaded.metadata.nx, loaded.metadata.ny) == (4, 4)
    assert (loaded.metadata.analysis_nx, loaded.metadata.analysis_ny) == (2, 2)


def test_load_odd_bayer_fits_crops_bottom_and_right_edges(tmp_path):
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    hdu = fits.PrimaryHDU(data)
    hdu.header["BAYERPAT"] = "BGGR"
    path = tmp_path / "odd_bayer.fits"
    hdu.writeto(path)

    loaded = load_flat(path)

    assert loaded.image.shape == (2, 2)
    assert (loaded.metadata.nx, loaded.metadata.ny) == (5, 5)


def test_unsupported_extension(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("not fits")
    with pytest.raises(FitsLoadError):
        load_flat(path)
