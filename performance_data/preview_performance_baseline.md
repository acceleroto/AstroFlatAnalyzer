# Preview performance baseline

Measured on 2026-08-02 using `performance_data/test_flat.fits`.

These are **single-threaded reference numbers**. The measurements ran in
isolated Python processes with direct synchronous calls to the numerical and
rendering functions. They do not include any multi-core or parallelism
improvement.

## Input

- File size: 52,223,040 bytes
- Source dimensions: 6252 × 4176
- Analysis dimensions: 3126 × 2088
- Format: mono Bayer flat, `RGGB`
- Bayer reduction: enabled, 2×2 averaged
- Array dtype: `float64`
- Finite-pixel fraction: 100%
- Smoothing: 100 source pixels, equivalent to 50 analysis pixels
- Display range: 80–100%
- Contours: enabled, 2% step

## Preview timing

Each preview computation timing is the median of three runs. Figure timing is
the median of two contour-enabled renders. Peak RSS was measured separately
for each target process and includes Python, FITS, NumPy, SciPy, and Matplotlib
overhead.

| Target | Reduction factor | Preview size | Load | Reduction | Preview math | Figure | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 900 px | 4 | 782 × 522 | 0.149 s | 0.057 s | 0.021 s | 0.045 s | 516.6 MiB |
| 1800 px | 2 | 1563 × 1044 | 0.171 s | 0.113 s | 0.190 s | 0.086 s | 666.5 MiB |

The full-resolution illumination map took 1.615 s and produced a 3126 × 2088
map.

## Fidelity against the full-resolution map

Errors are absolute illumination-percentage points. Contour-level agreement
means the effective contour-level arrays were identical.

| Target | Mean absolute error | 99th-percentile error | Maximum error | Contour levels |
|---:|---:|---:|---:|---|
| 900 px | 0.001029 | 0.021842 | 0.033421 | Match |
| 1800 px | 0.000500 | 0.000562 | 0.000615 | Match |

After upsampling the 900 px preview to the 1800 px preview dimensions, the
900-versus-1800 mean absolute difference was 0.002651 percentage points, with
a 99th-percentile difference of 0.021653 percentage points.

The current 1200 px target would use factor 3 for this file, producing an
approximately 1042 × 696 preview. The integer reduction factor means the
actual dimensions depend on the largest analysis-image dimension.
