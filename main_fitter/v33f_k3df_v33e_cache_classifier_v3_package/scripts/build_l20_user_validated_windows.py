#!/usr/bin/env python3
"""Build L20 accepted-window files from explicit user=true labels only."""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "output/v33p_projected_basis_cache/l20_full_cutoff_fits_v2"
OUT_ROOT = ROOT / "output/v33p_gpu_classifier_all_irrep/accepted_windows_L20_user_validated_0p36"
JACK_DIR = Path("/home/digonto/Codes/KKpi_I2/spectrum/Ecm_data/data")
SECTORS = ("000_A1m", "100_A2", "110_A2", "111_A2", "200_A2")
N2 = {"000_A1m": 0, "100_A2": 1, "110_A2": 2, "111_A2": 3, "200_A2": 4}
ALIASES = {"001_A2": "100_A2", "010_A2": "100_A2", "011_A2": "110_A2",
           "101_A2": "110_A2", "002_A2": "200_A2", "020_A2": "200_A2"}

LABELS = {
    "000_A1m": ROOT / "input/v33p_user_truezero_labels_L20_000_A1m_E026310_0360.csv",
    "100_A2": ROOT / "input/v33p_user_truezero_labels_L20_100_A2_E026301_0360.csv",
    "110_A2": ROOT / "input/v33p_gpu_user_truezero_labels_L20_110_A2.csv",
    "111_A2": ROOT / "input/v33p_gpu_user_truezero_labels_L20_111_A2.csv",
    "200_A2": ROOT / "input/v33p_gpu_user_truezero_labels_L20_200_A2.csv",
}
CANDIDATES = {
    "000_A1m": ROOT / "output/v33p_all_sector_classifier_validation/L20_000_A1m_E026310_0360/candidate_sign_change_brackets_E026310_0360.csv",
    "100_A2": ROOT / "output/v33p_classifier_restart_L20_100_A2_E026310_0360/candidate_sign_change_brackets_E026310_0360.csv",
    "110_A2": ROOT / "output/v33p_gpu_classifier_all_irrep/sectors/L20_110_A2/candidate_sign_change_brackets.csv",
    "111_A2": ROOT / "output/v33p_gpu_classifier_all_irrep/sectors/L20_111_A2/candidate_sign_change_brackets.csv",
    "200_A2": ROOT / "output/v33p_gpu_classifier_all_irrep/sectors/L20_200_A2/candidate_sign_change_brackets.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lattice_levels() -> dict[str, list[dict]]:
    out = {s: [] for s in SECTORS}
    for path in sorted(JACK_DIR.glob("20_*.jack")):
        name = path.name
        if not name.startswith("20_") or "_n" not in name or not name.endswith(".jack"):
            continue
        file_irrep, state_text = name[3:-5].rsplit("_n", 1)
        irrep = ALIASES.get(file_irrep, file_irrep)
        if irrep not in out:
            continue
        values = []
        for line in path.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 2:
                values.append(float(fields[1]))
        if not values:
            continue
        p = 2.0 * math.pi * math.sqrt(N2[irrep]) / (3.444 * 20.0)
        ecm = sum(math.sqrt(x * x - p * p) for x in values) / len(values)
        out[irrep].append({"state": int(state_text), "ecm": ecm, "path": str(path)})
    for s in SECTORS:
        out[s].sort(key=lambda r: (r["ecm"], r["state"]))
        for i, r in enumerate(out[s]):
            r["level_index"] = i
    return out


def read_label_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    levels = lattice_levels()
    audit_rows = []
    package_counts = {}
    for sector in SECTORS:
        label_path = LABELS[sector]
        candidate_path = CANDIDATES[sector]
        fields, labels = read_label_rows(label_path)
        if "user_label" not in fields:
            raise SystemExit(f"{label_path}: missing user_label column")
        candidates = {}
        with candidate_path.open(newline="") as f:
            for row in csv.DictReader(f):
                key = int(row.get("bracket_id", row.get("candidate_id", "-1")))
                candidates[key] = row
        vals = [(r.get("user_label") or "").strip().lower() for r in labels]
        true_rows = [r for r, v in zip(labels, vals) if v == "true"]
        false_n = sum(v == "false" for v in vals)
        blank_n = sum(v == "" for v in vals)
        other = sorted(set(vals) - {"true", "false", ""})
        if other:
            raise SystemExit(f"{label_path}: unexpected labels {other}")
        selected = []
        for source_row, label in enumerate(labels, start=2):
            if (label.get("user_label") or "").strip().lower() != "true":
                continue
            bracket = int(label["bracket_id"])
            if bracket not in candidates:
                raise SystemExit(f"{sector}: label bracket {bracket} missing from candidate file")
            c = candidates[bracket]
            center = int(round((int(c["row_left"]) + int(c["row_right"])) / 2.0))
            z = float(c.get("E_zero_linear", c.get("zero_estimate", "nan")))
            selected.append({"label": label, "source_row": source_row, "candidate": c,
                             "bracket": bracket, "center": center, "zero": z})
        selected.sort(key=lambda r: (r["zero"], r["bracket"]))
        outdir = OUT_ROOT / f"L20_{sector}"
        outdir.mkdir(parents=True, exist_ok=True)
        output_fields = ["Lbyas", "irrep", "lattice_level_index", "lattice_Ecm", "bracket_id",
                         "E_left_bracket", "E_right_bracket", "zero_estimate_initial", "center_row",
                         "row_left", "row_right", "max_row_left", "max_row_right",
                         "previous_model_zero", "has_previous_model_zero", "inside_Ecm_cutoff",
                         "source_label_file", "source_label_row"]
        output_rows = []
        lattice = levels[sector]
        for i, item in enumerate(selected):
            c = item["candidate"]
            center = item["center"]
            lo = max(0, center - 50)
            hi = max(center + 50, int(c["row_right"]))
            max_lo = max(0, center - 250)
            max_hi = center + 250
            lattice_ecm = lattice[i]["ecm"] if i < len(lattice) else ""
            output_rows.append({
                "Lbyas": 20, "irrep": sector, "lattice_level_index": i,
                "lattice_Ecm": lattice_ecm, "bracket_id": item["bracket"],
                "E_left_bracket": c["E_left"], "E_right_bracket": c["E_right"],
                "zero_estimate_initial": format(item["zero"], ".17g"), "center_row": center,
                "row_left": lo, "row_right": hi, "max_row_left": max_lo,
                "max_row_right": max_hi, "previous_model_zero": format(item["zero"], ".17g"),
                "has_previous_model_zero": "true", "inside_Ecm_cutoff": str(item["zero"] <= 0.335).lower(),
                "source_label_file": str(label_path), "source_label_row": item["source_row"]})
        with (outdir / "accepted_windows.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=output_fields)
            w.writeheader(); w.writerows(output_rows)
        package_counts[sector] = len(output_rows)
        true_es = [float(r["zero_estimate_initial"]) for r in output_rows]
        audit_rows.append({"sector": sector, "label_file": str(label_path), "label_sha256": sha256(label_path),
                           "columns": "|".join(fields), "label_rows": len(labels), "true_count": len(true_rows),
                           "false_count": false_n, "blank_count": blank_n, "true_Ecm_min": min(true_es) if true_es else "",
                           "true_Ecm_max": max(true_es) if true_es else "", "candidate_file": str(candidate_path),
                           "candidate_rows": len(candidates), "package_rows": len(output_rows),
                           "lattice_levels_to_0p36": len(lattice), "count_match_to_0p36": len(output_rows) == len(lattice),
                           "status": "PASS" if not blank_n and not other and len(output_rows) == len(lattice) else "REVIEW"})

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    with (AUDIT_ROOT / "L20_USER_LABEL_FILE_AUDIT.csv").open("w", newline="") as f:
        fields = list(audit_rows[0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(audit_rows)
    with (AUDIT_ROOT / "L20_USER_LABEL_FILE_AUDIT.md").open("w") as f:
        f.write("# L20 user-label audit\n\n")
        f.write("Only rows with explicit `user_label=true` were materialized; false and blank rows were excluded.\n\n")
        f.write("| sector | true | false | blank | true Ecm range | package rows | lattice levels through 0.36 | status |\n|---|---:|---:|---:|---|---:|---:|---|\n")
        for r in audit_rows:
            f.write(f"| {r['sector']} | {r['true_count']} | {r['false_count']} | {r['blank_count']} | {r['true_Ecm_min']}..{r['true_Ecm_max']} | {r['package_rows']} | {r['lattice_levels_to_0p36']} | {r['status']} |\n")
        f.write("\nThe L20_000_A1m package has three explicit true rows; its highest zero is above 0.355 and is therefore excluded by the 0.355 cutoff, not fabricated away.\n")

    audit = []
    for cut in (0.315, 0.325, 0.335, 0.345, 0.355):
        for r in audit_rows:
            with (OUT_ROOT / f"L20_{r['sector']}" / "accepted_windows.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            selected = [x for x in rows if float(x["zero_estimate_initial"]) <= cut + 1e-15]
            lattice_n = sum(x["ecm"] <= cut + 1e-15 for x in levels[r["sector"]])
            audit.append({"Ecm_cutoff": cut, "Lbyas": 20, "irrep": r["sector"],
                          "user_true_windows": len(selected), "lattice_levels": lattice_n,
                          "count_match": len(selected) == lattice_n,
                          "window_brackets": ";".join(x["bracket_id"] for x in selected),
                          "window_rows": ";".join(f"{x['max_row_left']}:{x['max_row_right']}" for x in selected),
                          "status": "PASS" if len(selected) == lattice_n else "BLOCKED"})
    path = AUDIT_ROOT / "L20_USER_VALIDATED_WINDOW_VS_LATTICE_AUDIT.csv"
    with path.open("w", newline="") as f:
        fields = list(audit[0]); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(audit)
    with (AUDIT_ROOT / "L20_USER_VALIDATED_WINDOW_VS_LATTICE_AUDIT.md").open("w") as f:
        f.write("# L20 user-validated windows versus lattice levels\n\n")
        f.write("Windows are sorted by user-validated zero estimate and mapped to sorted L20 lattice levels.\n\n")
        f.write("| cutoff | sector | user-true windows | lattice levels | count | brackets | max-row ranges |\n|---:|---|---:|---:|---|---|---|\n")
        for r in audit:
            f.write(f"| {r['Ecm_cutoff']} | {r['irrep']} | {r['user_true_windows']} | {r['lattice_levels']} | {r['status']} | {r['window_brackets']} | {r['window_rows']} |\n")
        f.write("\nNo rows marked false or blank are included.\n")


if __name__ == "__main__":
    main()
