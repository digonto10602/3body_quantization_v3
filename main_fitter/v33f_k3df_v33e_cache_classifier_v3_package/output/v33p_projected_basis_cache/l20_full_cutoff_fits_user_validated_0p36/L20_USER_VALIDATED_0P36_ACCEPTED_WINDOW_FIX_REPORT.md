# L20 user-validated accepted-window and cache rebuild report

Status: `L20_USER_VALIDATED_PREFIT_BLOCKED_NO_FIT_RUN`

## Scope

- Lbyas: 20 only.
- L24 was not resumed or used.
- Explicit `user_label=true` rows were included; false and blank rows were excluded.
- No cachegen, original-cache regeneration, minimization, or original-cache fallback was run.

## User-label audit

| sector | true | false | blank | new package windows |
|---|---:|---:|---:|---:|
| 000_A1m | 3 | 7 | 0 | 3 |
| 100_A2 | 2 | 23 | 0 | 2 |
| 110_A2 | 4 | 32 | 0 | 4 |
| 111_A2 | 5 | 25 | 0 | 5 |
| 200_A2 | 5 | 30 | 0 | 5 |

The 000_A1m third true row is above 0.355 and is retained in the package. It is excluded by cutoff filtering, not discarded as false.

## Old versus new package/cache coverage

| sector | old windows | new user-true windows | old cache rows | new cache rows | build seconds |
|---|---:|---:|---:|---:|---:|
| 000_A1m | 1 | 3 | 501 | 1503 | 713.78 |
| 100_A2 | 2 | 2 | 1002 | 1002 | 299.81 |
| 110_A2 | 4 | 4 | 2004 | 2004 | 860.97 |
| 111_A2 | 2 | 5 | 1002 | 2505 | 1571.52 |
| 200_A2 | 3 | 5 | 1503 | 2505 | 1463.70 |

## Cache validation

All five new caches pass metadata/binary row-count, accepted-window hash, expanded-row coverage, direct-versus-basis determinant, and sign checks. Each used 18 requested and 18 actual OpenMP threads. The largest matrix difference was `4.65661e-10`, accepted under the explicit `1e-8` matrix roundoff tolerance; raw determinant absolute differences and sign agreement remain the primary checks.

## Fit gate

The prefit audit passes all rows except:

- cutoff `0.355`, `L20/200_A2`: lattice levels `4`, explicit user-true windows `5`.

Therefore the five-cutoff fit suite was not launched. No parameters, covariance, model spectrum, or chi-square are fabricated for the blocked suite.

Detailed audits:

- `L20_USER_LABEL_FILE_AUDIT.csv/.md`
- `L20_USER_VALIDATED_WINDOW_VS_LATTICE_AUDIT.csv/.md`
- `L20_PREFIT_COVERAGE_AUDIT_USER_VALIDATED_0P36.csv/.md`

