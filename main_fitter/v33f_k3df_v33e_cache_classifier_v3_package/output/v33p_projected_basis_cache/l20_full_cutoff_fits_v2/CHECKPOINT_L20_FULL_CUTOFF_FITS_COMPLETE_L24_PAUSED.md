# L20 full cutoff-fit checkpoint

## Status

`CHECKPOINT_STATUS = L20_FULL_CUTOFF_FITS_COMPLETE_L24_PAUSED`

This checkpoint preserves the L20 projected-basis hot-window cutoff suite before user-validated accepted-window expansion. L24 was not resumed by this task; its determinant processes were inactive when checked.

## Validated state

- L20 expanded projected-basis row coverage: PASS for `000_A1m`, `100_A2`, `110_A2`, `111_A2`, and `200_A2`.
- L20 projected-basis smoke tests: PASS for one-sector, two-sector, and all-five-sector configurations.
- Runnable cutoffs: `0.315`, `0.325`, `0.335`.
- Blocked cutoffs: `0.345`, `0.355`, because the promoted accepted-window package did not contain all newly selected lattice levels.
- Runnable fits used projected-basis hot-window mode with `fallback_full_scan=0` and no full scans.

## Reports

- `L20_FULL_CUTOFF_DEPENDENCE_FINAL_REPORT.md`
- `L20_FULL_ECM_CUTOFF_FIT_SUMMARY.md`
- `L20_LATTICE_VS_MODEL_ALL_CUTOFFS.md`
- `PREFIT_COVERAGE_AUDIT.md`
- `L20_ONLY_ROW_COVERAGE_AUDIT.md` in the parent projected-basis cache directory
- `L20_ONLY_PROJECTED_BASIS_FITTER_SMOKE_TEST.md` in the parent projected-basis cache directory

## Reproduction scope

The checkpoint includes the L20 cutoff reports, lattice/model provenance tables, covariance/error outputs, sensitivity outputs, fitter orchestration scripts, and user label inputs needed for the next accepted-window rebuild. Original external F3inv/Vsel caches, full projected-basis caches, raw scans, and build artifacts are intentionally excluded.

## Next step

Read the authoritative L20 user-label CSVs, regenerate accepted windows from rows explicitly marked `true`, rebuild only expanded hot-window projected-basis caches with row-level OpenMP, rerun coverage, and fit only after the new package passes audit. Keep L24 inactive.
