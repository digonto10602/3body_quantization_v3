# L20 user-validated 0.36 final report

Status: `L20_USER_VALIDATED_PREFIT_BLOCKED_NO_FIT_RUN`

- Checkpoint pushed before changes: `75ffff22273b854ce9479ed7c9019d0cbcff7631` with tag `checkpoint_L20_full_cutoff_fits_complete_L24_paused`.
- L20 only; L24 remained inactive.
- User labels were audited and only explicit true rows were materialized.
- All five expanded hot-window projected-basis caches passed row coverage, hash, matrix/determinant, and sign checks.
- OpenMP row-level generation used 18 requested/18 actual threads, with deterministic serial output.
- Prefit passes through 0.345.
- Prefit blocks 0.355 because `L20/200_A2` has 4 lattice levels and 5 explicit user-true windows.
- No minimization or fit suite was run.

The next action is to resolve the authoritative `L20/200_A2` user-true/lattice-level mapping before any fit is allowed.

