# Full export performance baseline

Measured on 2026-08-02 using `performance_data/test_flat.fits`.

The benchmark uses `performance_data/export_benchmark.py` and the development
environment. Each timed value is one warm-up followed by one measured
iteration; a separate release-pinned run is still required before publishing a
release build. Cold rows include full-resolution illumination-map smoothing.
Cached rows reuse that map and report the map stage as zero. PNG and JPEG
quality settings are unchanged from the application.

## Input and environment

- File size: 52,223,040 bytes
- Source dimensions: 6252 × 4176
- Analysis dimensions: 3126 × 2088
- Format: mono Bayer flat, `RGGB`
- Bayer reduction: enabled, 2×2 averaged
- Analysis dtype: `float64`
- Smoothing: 100 source pixels, equivalent to 50 analysis pixels
- Display range: 80–100%
- JPEG quality: 95
- Python 3.13.14, NumPy 2.5.1, SciPy 1.18.0, Matplotlib 3.11.1, Pillow 12.3.0
- CPU count: 12

## Full-map worker scaling

The exact halo-tiled Gaussian implementation was measured before each target's
rendering matrix:

| Export target | Worker 1 | Worker 4 | Worker 8 | Parallel speedup |
|---|---:|---:|---:|---:|
| Native | 1.584 s | 0.594 s | — | 2.67× at 4 workers |
| 50% (3126 × 2088) | 1.611 s | 0.578 s | 0.421 s | 3.83× at 8 workers |
| 1920w (1920 × 1282) | 1.559 s | 0.579 s | 0.404 s | 3.86× at 8 workers |

The native run was limited to workers 1 and 4 because the benchmark's repeated
native raster stages reached approximately 5 GiB high-water RSS. The
application retains only the illumination map, releases the resampled array
after encoding, and does not retain benchmark figures or encoded payloads.
Run native targets in a fresh process when collecting a complete worker
matrix.

## Export stages

Values below are representative cold/cache-hit measurements. `resample`,
`figure`, `draw`, and `encode` are in-memory stage medians; `write` measures
writing encoded bytes followed by an atomic sibling-file replacement. The
`render_save` value, also recorded in the JSON report, exercises the complete
production Agg render-and-save path.

| Target | Format | Contours | Map cold | Resample | Figure | Agg draw | Encode | Write | Output |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Native | PNG | off | 1.584 s | 0.312 s | 0.013 s | 2.209 s | 2.239 s | 0.0004 s | 6252×4176, 490161 B |
| Native | JPEG | off | 1.584 s | 0.301 s | 0.012 s | 1.770 s | 1.784 s | 0.0010 s | 6252×4176, 549278 B |
| 50% | PNG | on | 1.611 s | 0.001 s | 0.240 s | 0.707 s | 0.812 s | 0.0005 s | 3126×2088, 446953 B |
| 50% | JPEG | on | 1.611 s | 0.001 s | 0.248 s | 0.746 s | 0.709 s | 0.0003 s | 3126×2088, 311631 B |
| 50% | PNG | off | 1.611 s | 0.001 s | 0.006 s | 0.468 s | 0.568 s | 0.0003 s | 3126×2088, 214153 B |
| 50% | JPEG | off | 1.611 s | 0.001 s | 0.005 s | 0.446 s | 0.455 s | 0.0004 s | 3126×2088, 173348 B |
| 1920w | PNG | on | 1.559 s | 0.029 s | 0.093 s | 0.263 s | 0.315 s | 0.0003 s | 1920×1282, 251262 B |
| 1920w | JPEG | on | 1.559 s | 0.029 s | 0.088 s | 0.299 s | 0.272 s | 0.0002 s | 1920×1282, 169800 B |
| 1920w | PNG | off | 1.559 s | 0.029 s | 0.004 s | 0.166 s | 0.213 s | 0.0002 s | 1920×1282, 110666 B |
| 1920w | JPEG | off | 1.559 s | 0.029 s | 0.004 s | 0.171 s | 0.176 s | 0.0002 s | 1920×1282, 82277 B |

Cached exports remove the map stage while retaining the same resample,
render, encode, dimensions, and output bytes. The full JSON report records
both cache modes, output SHA-256 references, map SHA-256 references, peak RSS,
process CPU time, and maximum parallel-versus-serial map difference.

## Fidelity reference

- Preallocated `scipy.ndimage.zoom` output is bitwise identical to the prior
  allocation-and-copy path for finite and nonfinite odd-sized inputs.
- Serial and threaded full maps are compared with `rtol=0` and `atol=1e-12`,
  including NaN-aware comparisons.
- Contour levels are generated from the same export array and settings.
- PNG compression-only comparisons decode to identical RGBA pixels and retain
  exact dimensions.
- On the 1920w contour-off sample, compression levels 1/3/6 took 0.229/0.224/
  0.222 s and produced 265129/143563/110666 bytes; level 6 was both fastest
  and smallest in this run, so the existing default was retained.
- JPEG quality remains the configured Pillow quality value; decoded dimensions
  remain exact and JPEG differences are not treated as lossless.

To reproduce the complete matrix, run each large target in a fresh process:

```text
.venv/bin/python performance_data/export_benchmark.py path/to/flat.fits \
  --targets native --workers 1,2,4,8 --contours on,off \
  --formats png,jpeg --iterations 3 --json
```

The same command with the release environment created from
`constraints-release.txt` provides the release-pinned comparison.
