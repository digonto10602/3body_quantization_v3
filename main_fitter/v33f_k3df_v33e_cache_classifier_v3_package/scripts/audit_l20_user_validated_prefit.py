#!/usr/bin/env python3
"""Audit user-validated L20 windows and hot-window cache coverage by cutoff."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOW_ROOT = ROOT / "output/v33p_gpu_classifier_all_irrep/accepted_windows_L20_user_validated_0p36"
CACHE_ROOT = ROOT / "output/v33p_projected_basis_cache/hot_windows_L20_user_validated_0p36"
OUT = ROOT / "output/v33p_projected_basis_cache/l20_full_cutoff_fits_user_validated_0p36"
SECTORS = ("000_A1m", "100_A2", "110_A2", "111_A2", "200_A2")
CUTS = (0.315, 0.325, 0.335, 0.345, 0.355)


def lattice_counts() -> dict[float, dict[str, int]]:
    # Reuse the exact parser used by the prior L20 fit suite.
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_l20_full_cutoff_fits_v2 import lattice_records
    return {cut: {s: sum(r["irrep"] == s for r in lattice_records(cut)[0]) for s in SECTORS} for cut in CUTS}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lcounts = lattice_counts()
    rows_out = []
    for cut in CUTS:
        for s in SECTORS:
            wpath = WINDOW_ROOT / f"L20_{s}" / "accepted_windows.csv"
            cdir = CACHE_ROOT / f"L20_{s}"
            rows = list(csv.DictReader(wpath.open(newline=""))) if wpath.exists() else []
            selected = [r for r in rows if float(r["zero_estimate_initial"]) <= cut + 1e-15]
            requested = set()
            for r in selected:
                lo = int(r.get("max_row_left", r["row_left"]))
                hi = int(r.get("max_row_right", r["row_right"]))
                requested.update(range(lo, hi + 1))
            meta_path = cdir / "projected_basis_hotwindows.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            saved = set(meta.get("row_indices", []))
            missing = sorted(requested - saved)
            files_ok = all((cdir / x).exists() for x in ("projected_basis_hotwindows.bin", "projected_basis_hotwindows.json", "projected_basis_hotwindows_validation.csv", "PROJECTED_BASIS_HOTWINDOW_CACHE_REPORT.md"))
            hash_ok = bool(meta) and meta.get("accepted_windows_sha256") == __import__("hashlib").sha256(wpath.read_bytes()).hexdigest()
            binary_rows = 0
            if (cdir / "projected_basis_hotwindows_validation.csv").exists():
                binary_rows = max(0, sum(1 for _ in cdir.joinpath("projected_basis_hotwindows_validation.csv").open()) - 1)
            lattice_n = lcounts[cut][s]
            status = "PASS" if files_ok and hash_ok and not missing and len(selected) == lattice_n and binary_rows == int(meta.get("n_rows_saved", -1)) else "BLOCKED"
            rows_out.append({"Ecm_cutoff": cut, "Lbyas": 20, "irrep": s, "lattice_levels": lattice_n,
                             "user_true_windows": len(selected), "requested_window_rows": len(requested),
                             "cache_rows": len(saved), "binary_validation_rows": binary_rows,
                             "missing_rows": len(missing), "missing_first": missing[:5],
                             "required_cache_files": files_ok, "accepted_windows_hash_match": hash_ok,
                             "status": status})
    path = OUT / "L20_PREFIT_COVERAGE_AUDIT_USER_VALIDATED_0P36.csv"
    with path.open("w", newline="") as f:
        fields = list(rows_out[0]); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows_out)
    with (OUT / "L20_PREFIT_COVERAGE_AUDIT_USER_VALIDATED_0P36.md").open("w") as f:
        f.write("# L20 user-validated prefit coverage audit\n\n")
        f.write("Only explicit user=true windows and the new expanded projected-basis cache root are checked. No original-cache fallback is permitted.\n\n")
        f.write("| cutoff | sector | lattice | user true | requested rows | cache rows | missing | hash | status |\n|---:|---|---:|---:|---:|---:|---:|---|---|\n")
        for r in rows_out:
            f.write(f"| {r['Ecm_cutoff']} | {r['irrep']} | {r['lattice_levels']} | {r['user_true_windows']} | {r['requested_window_rows']} | {r['cache_rows']} | {r['missing_rows']} | {'yes' if r['accepted_windows_hash_match'] else 'no'} | {r['status']} |\n")
        blocked = [r for r in rows_out if r["status"] != "PASS"]
        f.write("\n## Gate\n\n")
        if blocked:
            f.write("**BLOCKED: no minimization was run.** The following cutoff/sector rows fail the lattice-level versus explicit user-true window count gate:\n\n")
            for r in blocked:
                f.write(f"- cutoff {r['Ecm_cutoff']}, {r['irrep']}: lattice={r['lattice_levels']}, user_true={r['user_true_windows']}, missing_cache_rows={r['missing_rows']}\n")
        else:
            f.write("All requested cutoff/sector rows pass; the fit suite may be launched by its dedicated runner.\n")


if __name__ == "__main__":
    main()
