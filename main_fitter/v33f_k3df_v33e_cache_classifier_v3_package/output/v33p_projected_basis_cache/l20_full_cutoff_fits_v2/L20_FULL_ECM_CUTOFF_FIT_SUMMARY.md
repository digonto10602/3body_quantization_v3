# L20 full Ecm-cutoff fit suite

L24 was not resumed by this task. All runnable cases use L20 accepted sectors and projected-basis hot-window caches only.

| cutoff | levels | model_found | chi2 | dof | chi2/dof | K3iso0 | K3iso1 | K3B | K3E | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.315 | 6 | 6/6 | 3.216121053961791e-05 | 2 | 1.6080605269808956e-05 | 229109.08255481743 | -972421.14060757775 | 347174.05548116955 | -1226756.7068845264 | PASS |
| 0.325 | 9 | 9/9 | 9.93155870003628e-05 | 5 | 1.9863117400072559e-05 | 203654.78660989026 | -972421.14060757775 | 347174.05548116955 | -1226756.7068845264 | PASS |
| 0.335 | 12 | 12/12 | 0.0001265527817944407 | 8 | 1.5819097724305087e-05 | 69163.847351260105 | -1097463.7746150994 | 347174.05548116943 | -1226756.7068845264 | PASS |
| 0.345 | 14 | 0/14 | nan | nan | nan | nan | nan | nan | nan | BLOCKED_PREFIT_COVERAGE |
| 0.355 | 17 | 0/17 | nan | nan | nan | nan | nan | nan | nan | BLOCKED_PREFIT_COVERAGE |

## Interpretation

The validated accepted-window package covers through 0.335. Cutoffs 0.345 and 0.355 are reported as blocked when extra lattice levels lack accepted windows; no fake roots are inserted. Among runnable cutoffs, inspect parameter drift and covariance together before production interpretation.
