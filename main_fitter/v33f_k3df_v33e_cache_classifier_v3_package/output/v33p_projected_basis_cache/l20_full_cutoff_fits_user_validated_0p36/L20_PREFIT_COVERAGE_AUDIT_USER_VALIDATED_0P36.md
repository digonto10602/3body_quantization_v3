# L20 user-validated prefit coverage audit

Only explicit user=true windows and the new expanded projected-basis cache root are checked. No original-cache fallback is permitted.

| cutoff | sector | lattice | user true | requested rows | cache rows | missing | hash | status |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 0.315 | 000_A1m | 1 | 1 | 501 | 1503 | 0 | yes | PASS |
| 0.315 | 100_A2 | 2 | 2 | 1002 | 1002 | 0 | yes | PASS |
| 0.315 | 110_A2 | 1 | 1 | 501 | 2004 | 0 | yes | PASS |
| 0.315 | 111_A2 | 0 | 0 | 0 | 2505 | 0 | yes | PASS |
| 0.315 | 200_A2 | 2 | 2 | 1002 | 2505 | 0 | yes | PASS |
| 0.325 | 000_A1m | 1 | 1 | 501 | 1503 | 0 | yes | PASS |
| 0.325 | 100_A2 | 2 | 2 | 1002 | 1002 | 0 | yes | PASS |
| 0.325 | 110_A2 | 3 | 3 | 1503 | 2004 | 0 | yes | PASS |
| 0.325 | 111_A2 | 1 | 1 | 501 | 2505 | 0 | yes | PASS |
| 0.325 | 200_A2 | 2 | 2 | 1002 | 2505 | 0 | yes | PASS |
| 0.335 | 000_A1m | 1 | 1 | 501 | 1503 | 0 | yes | PASS |
| 0.335 | 100_A2 | 2 | 2 | 1002 | 1002 | 0 | yes | PASS |
| 0.335 | 110_A2 | 4 | 4 | 2004 | 2004 | 0 | yes | PASS |
| 0.335 | 111_A2 | 2 | 2 | 1002 | 2505 | 0 | yes | PASS |
| 0.335 | 200_A2 | 3 | 3 | 1503 | 2505 | 0 | yes | PASS |
| 0.345 | 000_A1m | 1 | 1 | 501 | 1503 | 0 | yes | PASS |
| 0.345 | 100_A2 | 2 | 2 | 1002 | 1002 | 0 | yes | PASS |
| 0.345 | 110_A2 | 4 | 4 | 2004 | 2004 | 0 | yes | PASS |
| 0.345 | 111_A2 | 4 | 4 | 2004 | 2505 | 0 | yes | PASS |
| 0.345 | 200_A2 | 3 | 3 | 1503 | 2505 | 0 | yes | PASS |
| 0.355 | 000_A1m | 2 | 2 | 1002 | 1503 | 0 | yes | PASS |
| 0.355 | 100_A2 | 2 | 2 | 1002 | 1002 | 0 | yes | PASS |
| 0.355 | 110_A2 | 4 | 4 | 2004 | 2004 | 0 | yes | PASS |
| 0.355 | 111_A2 | 5 | 5 | 2505 | 2505 | 0 | yes | PASS |
| 0.355 | 200_A2 | 4 | 5 | 2505 | 2505 | 0 | yes | BLOCKED |

## Gate

**BLOCKED: no minimization was run.** The following cutoff/sector rows fail the lattice-level versus explicit user-true window count gate:

- cutoff 0.355, 200_A2: lattice=4, user_true=5, missing_cache_rows=0
