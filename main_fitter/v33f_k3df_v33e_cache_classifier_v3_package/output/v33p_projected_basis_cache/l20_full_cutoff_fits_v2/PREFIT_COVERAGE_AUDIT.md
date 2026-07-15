# L20 prefit coverage audit

Lattice inputs are read from `/home/digonto/Codes/KKpi_I2/spectrum/Ecm_data/data`; parser skips one header line and reads whitespace column 2 as `En_lab`, then converts to Ecm using the fitter's L=20, xi=3.444 momentum formula.

| cutoff | irrep | lattice levels | accepted windows | requested rows | missing rows | cache files | status |
|---:|---|---:|---:|---:|---:|---|---|
| 0.315 | 000_A1m | 1 | 1 | 501 | 0 | yes | PASS |
| 0.315 | 100_A2 | 2 | 2 | 1002 | 0 | yes | PASS |
| 0.315 | 110_A2 | 1 | 1 | 501 | 0 | yes | PASS |
| 0.315 | 111_A2 | 0 | 0 | 0 | 0 | yes | PASS |
| 0.315 | 200_A2 | 2 | 2 | 1002 | 0 | yes | PASS |
| 0.325 | 000_A1m | 1 | 1 | 501 | 0 | yes | PASS |
| 0.325 | 100_A2 | 2 | 2 | 1002 | 0 | yes | PASS |
| 0.325 | 110_A2 | 3 | 3 | 1503 | 0 | yes | PASS |
| 0.325 | 111_A2 | 1 | 1 | 501 | 0 | yes | PASS |
| 0.325 | 200_A2 | 2 | 2 | 1002 | 0 | yes | PASS |
| 0.335 | 000_A1m | 1 | 1 | 501 | 0 | yes | PASS |
| 0.335 | 100_A2 | 2 | 2 | 1002 | 0 | yes | PASS |
| 0.335 | 110_A2 | 4 | 4 | 2004 | 0 | yes | PASS |
| 0.335 | 111_A2 | 2 | 2 | 1002 | 0 | yes | PASS |
| 0.335 | 200_A2 | 3 | 3 | 1503 | 0 | yes | PASS |
| 0.345 | 000_A1m | 1 | 1 | 501 | 0 | yes | PASS |
| 0.345 | 100_A2 | 2 | 2 | 1002 | 0 | yes | PASS |
| 0.345 | 110_A2 | 4 | 4 | 2004 | 0 | yes | PASS |
| 0.345 | 111_A2 | 4 | 2 | 1002 | 0 | yes | BLOCKED |
| 0.345 | 200_A2 | 3 | 3 | 1503 | 0 | yes | PASS |
| 0.355 | 000_A1m | 2 | 1 | 501 | 0 | yes | BLOCKED |
| 0.355 | 100_A2 | 2 | 2 | 1002 | 0 | yes | PASS |
| 0.355 | 110_A2 | 4 | 4 | 2004 | 0 | yes | PASS |
| 0.355 | 111_A2 | 5 | 2 | 1002 | 0 | yes | BLOCKED |
| 0.355 | 200_A2 | 4 | 3 | 1503 | 0 | yes | BLOCKED |

Cutoffs 0.345 and 0.355 are blocked if lattice levels exceed the validated accepted-window package; no fit is run for such a cutoff.
