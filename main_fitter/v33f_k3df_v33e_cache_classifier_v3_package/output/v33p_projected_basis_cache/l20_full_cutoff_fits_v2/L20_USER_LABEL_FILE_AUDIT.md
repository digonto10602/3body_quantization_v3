# L20 user-label audit

Only rows with explicit `user_label=true` were materialized; false and blank rows were excluded.

| sector | true | false | blank | true Ecm range | package rows | lattice levels through 0.36 | status |
|---|---:|---:|---:|---|---:|---:|---|
| 000_A1m | 3 | 7 | 0 | 0.26932269955846955..0.35794609133611466 | 3 | 2 | REVIEW |
| 100_A2 | 2 | 23 | 0 | 0.29155609987916675..0.30044149754668775 | 2 | 3 | REVIEW |
| 110_A2 | 4 | 32 | 0 | 0.3067911009653987..0.3287180485505124 | 4 | 5 | REVIEW |
| 111_A2 | 5 | 25 | 0 | 0.3194218456477709..0.3501284288578908 | 5 | 5 | PASS |
| 200_A2 | 5 | 30 | 0 | 0.2862443472262731..0.35346329686416345 | 5 | 4 | REVIEW |

The L20_000_A1m package has three explicit true rows; its highest zero is above 0.355 and is therefore excluded by the 0.355 cutoff, not fabricated away.
