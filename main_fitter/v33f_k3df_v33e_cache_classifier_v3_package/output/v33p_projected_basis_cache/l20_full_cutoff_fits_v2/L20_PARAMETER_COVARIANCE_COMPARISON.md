# L20 covariance and error comparison

The covariance matrices are native Minuit HESSE outputs and are positive definite for all runnable cutoffs.

| cutoff | covariance eigenvalues | maximum absolute off-diagonal correlation | interpretation |
|---:|---|---:|---|
| 0.315 | 5.116e14, 5.727e15, 1.663e16, 1.203e20 | 0.99586 | severe parameter degeneracy; very large errors |
| 0.325 | approximately 4.0, 4.0, 4.0, 4.0 | 8.4e-16 | numerically diagonal HESSE result with errors near 2; treat errors cautiously |
| 0.335 | approximately 4.0, 4.0, 4.0, 4.0 | 5.3e-15 | numerically diagonal HESSE result with errors near 2; treat errors cautiously |

The near-exact 4/eigenvalue and 2/error pattern at 0.325 and 0.335 is recorded as observed output, not interpreted as evidence of physical parameter independence.


---

# L20 fit summary, Ecm_cutoff=0.315

- status: PASS
- projected-basis cache mode: hot_windows
- model_found: 6/6
- chi2: 3.2161210539617911e-05
- dof: 2
- chi2/dof: 1.6080605269808956e-05
- fallback/full_scan: 0/0
- Minuit covariance: positive_definite; eigenvalues: [511615980964070.0, 5727481407817757.0, 1.6628681450003592e+16, 1.2033310038678433e+20]
- fit evaluations: 91
- load time: 0.505836526 s
- total runtime: 1.0084960719977971 s

Per-level data: `lattice_vs_model_levels.csv`, `model_ecm_levels.csv`, and `lattice_vs_model_levels.md`.

---

# L20 fit summary, Ecm_cutoff=0.325

- status: PASS
- projected-basis cache mode: hot_windows
- model_found: 9/9
- chi2: 9.9315587000362794e-05
- dof: 5
- chi2/dof: 1.9863117400072559e-05
- fallback/full_scan: 0/0
- Minuit covariance: positive_definite; eigenvalues: [3.9999995231628875, 3.999999523162888, 3.9999995231628906, 3.9999995231629004]
- fit evaluations: 175
- load time: 0.611222727 s
- total runtime: 1.976086436989135 s

Per-level data: `lattice_vs_model_levels.csv`, `model_ecm_levels.csv`, and `lattice_vs_model_levels.md`.

---

# L20 fit summary, Ecm_cutoff=0.335

- status: PASS
- projected-basis cache mode: hot_windows
- model_found: 12/12
- chi2: 0.0001265527817944407
- dof: 8
- chi2/dof: 1.5819097724305087e-05
- fallback/full_scan: 0/0
- Minuit covariance: positive_definite; eigenvalues: [3.9999995231628462, 3.99999952316289, 3.999999523162892, 3.9999995231629084]
- fit evaluations: 90
- load time: 0.620488187 s
- total runtime: 2.6938688100053696 s

Per-level data: `lattice_vs_model_levels.csv`, `model_ecm_levels.csv`, and `lattice_vs_model_levels.md`.

---

# Ecm cutoff 0.345

Status: **BLOCKED_PREFIT_COVERAGE**. No model or fit values were fabricated.

---

# Ecm cutoff 0.355

Status: **BLOCKED_PREFIT_COVERAGE**. No model or fit values were fabricated.
