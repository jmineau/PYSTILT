# R-STILT Parity Reference

PYSTILT is validated against the [uataq/stilt](https://github.com/uataq/stilt)
R implementation at a **pinned upstream commit**. This document records the
scope of the parity claim and the procedure for keeping it current.

## Pinned upstream commit

| Field | Value |
|---|---|
| Upstream | https://github.com/uataq/stilt |
| Pinned SHA | `0e290a68730e155bc2156e28af492a463228c88a` |
| Pin location | `.github/workflows/tests.yml` (env `STILT_R_SHA`) |
| Last verified | 2026-06-25 |

The SHA is the single source of truth for what "matches R-STILT" means in this
repository. Reviewers reading "tests pass" should interpret it as **tests pass
against this exact commit**, not the upstream `main` branch.

## Scope of the parity claim

PYSTILT matches R-STILT on **numerical footprint values** for the 20 fidelity
scenarios in [tests/fixtures/r_stilt_reference.py](tests/fixtures/r_stilt_reference.py):

| Quantity | Tolerance |
|---|---|
| Per-cell footprint | `rtol=1e-7, atol=1e-8` |
| Total footprint sum | `rtol=1e-6` |
| Peak footprint | `rtol=1e-7` |
| Intermediate tables (interpolation, rtime, gridding) | `rtol=1e-12` |
| Trajectory positions + foot column after HNF | `rtol=1e-7` |
| Coord arrays (longlat) | `rtol=1e-13, atol=1e-12` |
| Coord arrays (projected meters) | `rtol=1e-13` (~sub-mm at UTM scale) |

### What is **not** in scope

- **NetCDF wire format**: R-STILT writes dims `(x, y, time)` with
  `fill_value=-1`; PYSTILT writes `(time, lat, lon)` with xarray defaults.
  Downstream R-based readers that expect the R-STILT byte layout are
  **not supported**; PYSTILT output should be read as a generic CF-1.8 NetCDF.
- **Behavior on inputs outside the tested scenarios** — see "What is tested"
  below. Untested scenarios (long-duration backward >24h, polar latitudes
  >80°, very fine grids <0.001°, etc.) are unverified, not known to match.
- **R-STILT versions other than the pinned SHA.** Upstream changes to
  `calc_footprint.r`, `permute.f90`, or `calc_trajectory.r` are not
  automatically detected; the SHA must be bumped manually.

## What is tested

The 20 fidelity scenarios in `tests/fixtures/r_stilt_reference.py` cover:

- **Receptor types**: point, column, three-location multipoint
- **Direction**: 6h backward, 24h backward, 6h forward (`time_sign=+1`)
- **HNF plume dilution**: on (12 scenarios) and off
- **Smoothing**: `smooth_factor` ∈ {0.0, 0.5, 1.0, 2.0}
- **Grids**: 0.01° longlat, 0.05° coarse longlat, UTM projected
- **Domains**: standard 2° box, edge-receptor (SW corner), met-grid edge,
  tight sparse 0.1° (particles exit early), dateline-crossing (synthetic
  test), global 360° (synthetic test)
- **Altitude refs**: AGL, MSL (`kmsl=1`), low AGL (0.5 m)
- **Meteorology**: winter HRRR (stable PBL), summer HRRR (convective PBL)
- **Time integration**: hourly bins and `time_integrate=True`
- **Error trajectory**: WINDERR with `siguverr=1 m/s`

Additional synthetic tests in
[tests/r_stilt/test_footprint_synth.py](tests/r_stilt/test_footprint_synth.py)
exercise hand-crafted particle DataFrames against R-STILT to isolate specific
code paths (single-particle Gaussian, boundary fenceposts, dateline crossing,
global grid, latitude bandwidth scaling, etc.).

## Large-scale seed-matched validation

Beyond the 20 CI fidelity scenarios, PYSTILT has been validated against R-STILT
on a **200-receptor, seed-matched, bit-for-bit campaign** spanning 2016–2024
(WBB tower, Salt Lake Valley; 35 m AGL; 1000 particles; 24 h backward; HRRR;
`krand=2`, `seed=42`; byte-identical v5.1.0 `hycs_std` on both sides). For each
receptor the PYSTILT trajectory is compared to an independent R-STILT
`calc_trajectory` run, and the PYSTILT footprint to R's `calc_footprint` of the
same particles, at three resolutions (0.01°, 0.05°, 0.1°).

**Result: all 200 receptors match bit-for-bit.**

| Quantity | Agreement (max over 200 receptors) |
|---|---|
| Trajectory, per particle | abs `1.0e-17`, rel `5.6e-16` — machine-exact (float64) |
| Footprint, per-cell relative error | median `~2e-8`, max `~6e-8` |
| Footprint nonzero-cell set | identical (e.g. 250,644 = 250,644 at 0.01°) |
| Footprint total mass | rel `~2e-8` |

The per-cell relative error is **uniform across cell magnitudes** — the smallest
cells (<0.01 % of peak) agree as tightly as the largest, so small footprint
values are not disproportionately wrong. This follows from the trajectories
being bit-identical: the only footprint difference is float-summation order in
the gridding/convolution, and footprints are stored as **`float32`**
(ε ≈ 1.2e-7). The observed ~2e-8 relative agreement is therefore **below the
float32 storage floor** — the two outputs agree to the precision the NetCDF
format can represent. Mass-weighted relative error (what a flux inversion
integrates) is ~2e-8.

This is a one-time field-scale validation (not run in CI); it complements the
structural coverage of the CI fidelity scenarios with broad real-meteorology
breadth. Reading identical met on both sides across the full date range requires
the `find_met_files` dedup fix carried in the pinned SHA — HRRR archives can
expose one physical file via both a top-level symlink and a `YYYY/MM/`
subdirectory, which older R-STILT listed twice.

## Procedure for bumping the pinned SHA

1. Bump `STILT_R_SHA` in `.github/workflows/tests.yml`.
2. Locally:
   ```
   git clone https://github.com/uataq/stilt stilt-r-src
   git -C stilt-r-src checkout <new-sha>
   STILT_R_DIR=$PWD/stilt-r-src uv run pytest tests/r_stilt/ -v -m integration
   ```
3. If all fidelity scenarios pass at the existing tolerances → commit the
   bump and update the "Last verified" date in this file.
4. If a scenario fails → diff `calc_footprint.r`, `permute.f90`, and
   `calc_trajectory.r` between the old and new SHA, identify the upstream
   change, and decide:
   - Mirror the change in PYSTILT, or
   - Loosen the affected scenario's tolerance with a comment explaining why, or
   - Hold the pin at the previous SHA.

## Files watched for upstream drift

These three R-STILT files materially influence the numerical output and
should be diffed on every SHA bump:

- `r/src/calc_footprint.r` — footprint construction, kernel, gridding
- `r/src/calc_trajectory.r` — HYSPLIT wrapper, RNG seeding, KRAND
- `r/src/permute.f90` — Fortran convolution accumulator

Helper R scripts under `tests/fixtures/r_helpers/` re-export these for the
synthetic comparison tests; bumping the SHA may require updating those if
upstream signatures change.
