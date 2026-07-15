# L20 full cutoff-dependence final report

## Scope and checkpoint

- Lbyas: `20` only
- Sectors: `000_A1m`, `100_A2`, `110_A2`, `111_A2`, `200_A2`
- Projected-basis cache mode: `hot_windows`
- fallback/full_scan: `0/0` for runnable fits
- L24 status: no active PIDs remained when the requested stop was issued; no L24 work was launched or resumed by this task.

## Prefit coverage

See `PREFIT_COVERAGE_AUDIT.csv/.md`. Cache files required per sector were checked, and requested `max_row_left..max_row_right` sets were compared with JSON `row_indices`.

Exact lattice input directory: `/home/digonto/Codes/KKpi_I2/spectrum/Ecm_data/data`. The fitter parser reads `20_<file_irrep>_n<state>.jack`, skips one header line, reads whitespace column 2 as `En_lab`, converts to Ecm with xi=3.444, filters by mean Ecm cutoff, and sorts by sector order and state.

## Per-cutoff results

### Ecm_cutoff=0.315

- levels/model_found: `6` / `6/6`
- chi2, dof, chi2/dof: `3.216121053961791e-05`, `2`, `1.6080605269808956e-05`
- fallback/full_scan: `0/0`
- parameters: K3iso0=`229109.08255481743` +/- `372679103.99476475`, K3iso1=`-972421.14060757775` +/- `314679287.94938761`, K3B=`347174.05548116955` +/- `334353772.34994745`, K3E=`-1226756.7068845264` +/- `10954737009.701321`
- model levels: `Ecm_cut_0p315/model_ecm_levels.csv`
- lattice-vs-model: `Ecm_cut_0p315/lattice_vs_model_levels.csv`
- sensitivity/impact: `Ecm_cut_0p315/parameter_sensitivity.csv`, `Ecm_cut_0p315/parameter_impact_summary.csv`

### Ecm_cutoff=0.325

- levels/model_found: `9` / `9/9`
- chi2, dof, chi2/dof: `9.93155870003628e-05`, `5`, `1.9863117400072559e-05`
- fallback/full_scan: `0/0`
- parameters: K3iso0=`203654.78660989026` +/- `1.9999998807907211`, K3iso1=`-972421.14060757775` +/- `1.9999998807907189`, K3B=`347174.05548116955` +/- `1.9999998807907189`, K3E=`-1226756.7068845264` +/- `1.9999998807907191`
- model levels: `Ecm_cut_0p325/model_ecm_levels.csv`
- lattice-vs-model: `Ecm_cut_0p325/lattice_vs_model_levels.csv`
- sensitivity/impact: `Ecm_cut_0p325/parameter_sensitivity.csv`, `Ecm_cut_0p325/parameter_impact_summary.csv`

### Ecm_cutoff=0.335

- levels/model_found: `12` / `12/12`
- chi2, dof, chi2/dof: `0.0001265527817944407`, `8`, `1.5819097724305087e-05`
- fallback/full_scan: `0/0`
- parameters: K3iso0=`69163.847351260105` +/- `1.9999998807907151`, K3iso1=`-1097463.7746150994` +/- `1.9999998807907151`, K3B=`347174.05548116943` +/- `1.9999998807907211`, K3E=`-1226756.7068845264` +/- `1.9999998807907189`
- model levels: `Ecm_cut_0p335/model_ecm_levels.csv`
- lattice-vs-model: `Ecm_cut_0p335/lattice_vs_model_levels.csv`
- sensitivity/impact: `Ecm_cut_0p335/parameter_sensitivity.csv`, `Ecm_cut_0p335/parameter_impact_summary.csv`

### Ecm_cutoff=0.345

Status: **BLOCKED_PREFIT_COVERAGE**. Lattice levels exceeded accepted-window coverage; no fit, roots, or parameters were fabricated.

### Ecm_cutoff=0.355

Status: **BLOCKED_PREFIT_COVERAGE**. Lattice levels exceeded accepted-window coverage; no fit, roots, or parameters were fabricated.

## Interpretation and caveats

The highest runnable cutoff is operationally preferable only if parameter drift and covariance are acceptable. These remain L20-only fits, not a global production fit. Cutoffs 0.345 and 0.355 require additional accepted classifier windows/cache coverage for the extra lattice levels before they can be fitted.

Next action: review the blocked high-cutoff sectors and decide whether to generate/validate additional accepted windows; L24 should remain inactive until explicitly resumed.

## Covariance and parameter-impact interpretation

- The 0.315 covariance is positive definite but strongly degenerate: maximum absolute off-diagonal correlation is approximately 0.99586 and parameter errors are enormous.
- The 0.325 and 0.335 HESSE matrices are positive definite with eigenvalues approximately 4 and errors approximately 2, while correlations are numerically near zero. This exact pattern is treated as a Minuit/HESSE diagnostic caveat, not physical proof of parameter independence.
- Sensitivity files use finite-difference steps `(1000, 10000, 1000, 10000)` and retain model-found status. At all runnable cutoffs the most affected mapped model level is level 3 in the generated sensitivity summary; exact per-level derivatives are in each `parameter_sensitivity.csv`.
- At 0.335, representative maximum absolute derivatives are approximately 0.3068 for K3iso0, K3iso1, K3B, and K3E under their respective steps. Parameter-impact tables report the corresponding delta-chi2 values.

## Lattice/model provenance and reconstruction

Each runnable cutoff contains exact `lattice_input_files.csv`, `lattice_levels_used.csv`, `lattice_levels_excluded.csv`, `model_ecm_levels.csv`, `model_ecm_levels.json`, and `lattice_vs_model_levels.csv/.md`. The per-level correlated contribution is defined as `residual_i * (C^-1 residual)_i`; at 0.335 its reconstructed sum is `0.00012655278179443788`, agreeing with reported chi2 `0.0001265527817944407` within numerical precision.
