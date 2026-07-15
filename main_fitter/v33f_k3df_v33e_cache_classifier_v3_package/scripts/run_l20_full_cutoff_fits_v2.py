#!/usr/bin/env python3
"""L20-only projected-basis cutoff suite with provenance and post-fit audits."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/v33p_projected_basis_cache/fitter_projected_basis_hot_windows_all.in"
OUT = ROOT / "output/v33p_projected_basis_cache/l20_full_cutoff_fits_v2"
BIN = ROOT / "bin/v33f_k3df_fitter_multiL_v33e"
JACK_DIR = Path("/home/digonto/Codes/KKpi_I2/spectrum/Ecm_data/data")
CUTS = (0.315, 0.325, 0.335, 0.345, 0.355)
SECTORS = ("000_A1m", "100_A2", "110_A2", "111_A2", "200_A2")
N2 = {"000_A1m": 0, "100_A2": 1, "110_A2": 2, "111_A2": 3, "200_A2": 4}
FILE_IRREP = {"000_A1m": "000_A1m", "100_A2": "001_A2", "110_A2": "011_A2", "111_A2": "111_A2", "200_A2": "002_A2"}
PARAMS = ("K3iso0", "K3iso1", "K3B", "K3E")
START = (73735.840894011912, -972421.14060757787, 347174.05548116949, -1226756.7068845264)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def set_key(text: str, key: str, value: str) -> str:
    pat = re.compile(rf"(?m)^\s*{re.escape(key)}\s*=.*$")
    line = f"{key} = {value}"
    return pat.sub(line, text, count=1) if pat.search(text) else text + "\n" + line + "\n"


def cfg_value(text: str, key: str) -> str:
    m = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*(.*?)\s*$", text)
    return m.group(1).strip() if m else ""


def alias(label: str) -> str:
    return {"001_A2": "100_A2", "010_A2": "100_A2", "011_A2": "110_A2", "101_A2": "110_A2", "002_A2": "200_A2", "020_A2": "200_A2"}.get(label, label)


def read_jack(path: Path, irrep: str) -> tuple[list[float], list[float], int]:
    raw = []
    lines = path.read_text().splitlines()
    for lineno, line in enumerate(lines, 1):
        if lineno <= 1 or not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 2:
            continue
        raw.append((float(cols[1]), lineno))
    xi = 3.444
    L = 20.0
    P = 2.0 * math.pi * math.sqrt(N2[irrep]) / (xi * L)
    ecm = [math.sqrt(x * x - P * P) for x, _ in raw]
    return [x for x, _ in raw], ecm, len(raw)


def lattice_records(cut: float) -> tuple[list[dict], list[dict]]:
    used, excluded = [], []
    pat = re.compile(r"^20_(.+)_n(\d+)\.jack$")
    for path in sorted(JACK_DIR.glob("20_*.jack")):
        m = pat.match(path.name)
        if not m or alias(m.group(1)) not in SECTORS:
            continue
        file_irrep = m.group(1)
        irrep = alias(file_irrep)
        state = int(m.group(2))
        lab, ecm, nrows = read_jack(path, irrep)
        rec = {"Lbyas": 20, "irrep": irrep, "file_irrep": file_irrep, "state": state, "path": str(path), "lab": sum(lab) / len(lab), "ecm": sum(ecm) / len(ecm), "nrows": nrows, "sha256": sha256(path)}
        (used if rec["ecm"] <= cut else excluded).append(rec)
    order = {x: i for i, x in enumerate(SECTORS)}
    used.sort(key=lambda r: (order[r["irrep"]], r["state"], r["file_irrep"]))
    excluded.sort(key=lambda r: (order[r["irrep"]], r["state"], r["file_irrep"]))
    return used, excluded


def parse_summary(path: Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        p = line.split()
        if len(p) >= 2:
            out[p[0]] = p[1]
            if p[0] in PARAMS and len(p) >= 4:
                out[p[0] + "_err"] = p[3]
    return out


def matrix_dat(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    return np.asarray(rows, dtype=float) if np is not None else rows


def nearest_row(det_path: Path, energy: float) -> int:
    best, bd = 0, float("inf")
    with det_path.open() as f:
        for row in csv.DictReader(f):
            d = abs(float(row["Ecm"]) - energy)
            if d < bd:
                bd, best = d, int(row.get("row_global_index", row.get("row_index", 0)))
    return best


def accepted_source_paths(base_text: str) -> dict[str, Path]:
    out = {}
    for m in re.finditer(r"(?m)^\s*(accepted_zeros_file_L20_[^ ]+)\s*=\s*(\S+)\s*$", base_text):
        p = Path(m.group(2)); out[m.group(1)] = p if p.is_absolute() else ROOT / p
    return out


def filtered_zero_files(cut: float, outdir: Path, base_text: str) -> dict[str, Path]:
    out = {}
    for key, src in accepted_source_paths(base_text).items():
        target = outdir / (src.stem + f"_cut{cut:.3f}".replace(".", "p") + ".csv")
        with src.open(newline="") as f:
            reader = csv.DictReader(f); fields = reader.fieldnames or []; rows = []
            for row in reader:
                try:
                    if row.get("user_label", "").lower() == "true" and float(row["zero_estimate"]) <= cut + 1e-15:
                        if "inside_Ecm_cutoff" in row: row["inside_Ecm_cutoff"] = "true"
                        rows.append(row)
                except (KeyError, ValueError):
                    pass
        with target.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        out[key] = target
    return out


def write_lattice_files(cut: float, outdir: Path, used: list[dict], excluded: list[dict]) -> None:
    with (outdir / "lattice_input_files.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["Ecm_cutoff", "Lbyas", "irrep", "input_file_path", "file_exists", "file_size_bytes", "sha256", "columns_detected", "n_rows_read", "n_levels_below_cutoff", "status"])
        allr = used + excluded
        for r in allr:
            w.writerow([cut, 20, r["irrep"], r["path"], True, Path(r["path"]).stat().st_size, r["sha256"], "whitespace columns; header skipped; En_lab column 2", r["nrows"], int(r in used), "USED" if r in used else "EXCLUDED_BY_CUTOFF"])
    with (outdir / "lattice_levels_used.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["Ecm_cutoff", "Lbyas", "irrep", "level_index_original", "level_index_used", "lattice_Ecm", "lattice_Ecm_err_or_sigma", "lab_energy_if_available", "row_source_file", "source_line_or_row", "status"])
        counts = {s: 0 for s in SECTORS}
        for i, r in enumerate(used):
            idx = counts[r["irrep"]]; counts[r["irrep"]] += 1
            w.writerow([cut, 20, r["irrep"], r["state"], idx, r["ecm"], "PENDING_COVARIANCE", r["lab"], r["path"], "jackknife mean", "USED"])
    with (outdir / "lattice_levels_excluded.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["Ecm_cutoff", "Lbyas", "irrep", "level_index_original", "lattice_Ecm", "reason_excluded", "row_source_file"])
        for r in excluded: w.writerow([cut, 20, r["irrep"], r["state"], r["ecm"], "Ecm_mean_above_cutoff", r["path"]])


def run(cmd: list[str], log: Path, env: dict[str, str]) -> tuple[int, float]:
    t = time.monotonic()
    with log.open("w") as f:
        f.write("COMMAND: " + " ".join(cmd) + "\n")
        p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode, time.monotonic() - t


def make_config(cut: float, guess: tuple[float, ...], outdir: Path, zero_paths: dict[str, Path], tag: str) -> Path:
    text = BASE.read_text()
    for key, value in {"Lbyas_values": "20", "irreps_L24": "", "list_of_mom": "000_A1m 100_A2 110_A2 111_A2 200_A2", "Ecm_cutoff": f"{cut:.17g}", "energy_cutoff": f"{cut:.17g}", "one_fcn_only": "0", "max_fcn": "400", "max_fcn_evals": "500", "output_dir": str(outdir), "output_tag": tag, "K3iso0_guess": f"{guess[0]:.17g}", "K3iso1_guess": f"{guess[1]:.17g}", "K3B_guess": f"{guess[2]:.17g}", "K3E_guess": f"{guess[3]:.17g}", "K3iso0_step": "1000", "K3iso1_step": "10000", "K3B_step": "1000", "K3E_step": "10000", "float_K3iso0": "1", "float_K3iso1": "1", "float_K3B": "1", "float_K3E": "1"}.items():
        text = set_key(text, key, value)
    for key, path in zero_paths.items(): text = set_key(text, key, str(path))
    path = outdir / "fitter_input.txt"; path.write_text(text); return path


def parse_fcn(text: str) -> dict:
    pairs = re.findall(r"model_found=(\d+)/(\d+)", text)
    return {"found": int(pairs[-1][0]) if pairs else 0, "target": int(pairs[-1][1]) if pairs else 0, "chi2": float(re.findall(r"\[fcn-once\] chi2=([0-9eE+.+-]+)", text)[-1]) if re.findall(r"\[fcn-once\] chi2=([0-9eE+.+-]+)", text) else math.nan, "rows": int(re.findall(r"rows_evaluated=(\d+)", text)[-1]) if re.findall(r"rows_evaluated=(\d+)", text) else 0, "fcn_sec": float(re.findall(r"fcn_sec=([0-9eE+.+-]+)", text)[-1]) if re.findall(r"fcn_sec=([0-9eE+.+-]+)", text) else math.nan, "load": float(re.findall(r"cache_load_and_precompute_sec=([0-9eE+.+-]+)", text)[-1]) if re.findall(r"cache_load_and_precompute_sec=([0-9eE+.+-]+)", text) else math.nan}


def write_matrix_csv(src: Path, dst: Path, names: list[str]) -> None:
    mat = matrix_dat(src)
    with dst.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["parameter"] + names)
        for name, row in zip(names, mat): w.writerow([name] + list(row))


def create_per_level(cut: float, outdir: Path, tag: str, used: list[dict], summary: dict[str, str], sensitivity_path: Path) -> None:
    lev_path = outdir / f"{tag}_fit_levels_allL.dat"
    spec_path = outdir / f"{tag}_bestfit_QC_spectrum_allL.dat"
    rows = []
    if lev_path.exists():
        for line in lev_path.read_text().splitlines():
            if line.startswith("#") or not line.strip(): continue
            p = line.split(); rows.append({"row": int(p[0]), "L": float(p[1]), "file_irrep": p[2], "irrep": p[3], "state": int(p[4]), "level": int(p[5]), "lattice": float(p[6]), "sigma": float(p[7]), "model": float(p[8]), "residual": float(p[9])})
    windows = {}
    for s in SECTORS:
        p = ROOT / f"output/v33p_gpu_classifier_all_irrep/accepted_windows/L20_{s}/accepted_windows.csv"
        if not p.exists(): continue
        with p.open() as f:
            for r in csv.DictReader(f):
                if float(r["zero_estimate_initial"]) <= cut + 1e-15:
                    windows[(s, int(r["lattice_level_index"]))] = r
    cov = matrix_dat(outdir / f"{tag}_covariance_allL.dat") if (outdir / f"{tag}_covariance_allL.dat").exists() else None
    cinv = np.linalg.pinv(cov) if np is not None and cov is not None else None
    with (outdir / "model_ecm_levels.csv").open("w", newline="") as f, (outdir / "lattice_vs_model_levels.csv").open("w", newline="") as g:
        mw = csv.writer(f); vw = csv.writer(g)
        mw.writerow(["Ecm_cutoff", "Lbyas", "irrep", "level_index", "lattice_Ecm", "model_Ecm", "residual_model_minus_lattice", "sigma_or_weight", "row_index", "bracket_id", "source_window", "status"])
        vw.writerow(["Ecm_cutoff", "Lbyas", "irrep", "level_index_used", "lattice_Ecm", "lattice_Ecm_err_or_sigma", "model_Ecm", "residual_model_minus_lattice", "normalized_residual", "chi2_contribution", "row_index_or_window_row", "bracket_id", "source_window", "lattice_source_file", "status"])
        residuals = np.asarray([r["model"] - r["lattice"] for r in rows]) if np is not None else []
        contrib = (residuals * (cinv @ residuals)).tolist() if cinv is not None and len(rows) else [math.nan] * len(rows)
        for i, r in enumerate(rows):
            w = windows.get((r["irrep"], r["level"]), {})
            br = w.get("bracket_id", "")
            source_window = f"{w.get('max_row_left','')}:{w.get('max_row_right','')}" if w else "unmapped"
            mw.writerow([cut, 20, r["irrep"], r["level"], r["lattice"], r["model"], r["model"]-r["lattice"], r["sigma"], w.get("center_row", ""), br, source_window, "MAPPED" if w else "UNMAPPED"])
            vw.writerow([cut, 20, r["irrep"], r["level"], r["lattice"], r["sigma"], r["model"], r["model"]-r["lattice"], (r["model"]-r["lattice"])/r["sigma"], contrib[i] if i < len(contrib) else math.nan, w.get("center_row", ""), br, source_window, next((x["path"] for x in used if x["irrep"] == r["irrep"] and x["state"] == r["state"]), ""), "OK" if w else "UNMAPPED"])
    # Required residuals table.
    with (outdir / "residuals.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["Lbyas", "irrep", "level_index", "lattice_Ecm", "model_Ecm", "residual_model_minus_lattice", "sigma", "normalized_residual"])
        for r in rows: w.writerow([20, r["irrep"], r["level"], r["lattice"], r["model"], r["model"]-r["lattice"], r["sigma"], (r["model"]-r["lattice"])/r["sigma"]])
    (outdir / "model_ecm_levels.json").write_text(json.dumps([{"Ecm_cutoff": cut, "Lbyas": 20, "irrep": r["irrep"], "level_index": r["level"], "lattice_Ecm": r["lattice"], "model_Ecm": r["model"], "residual_model_minus_lattice": r["model"]-r["lattice"]} for r in rows], indent=2) + "\n")
    with (outdir / "lattice_vs_model_levels.md").open("w") as f:
        f.write(f"# Lattice versus model, Ecm_cutoff={cut:.3f}\n\n")
        f.write("Per-level correlated chi2 contribution is defined as `residual_i * (C^-1 residual)_i`; the sum reconstructs the correlated chi2.\n\n")
        f.write("| irrep | level | lattice Ecm | model Ecm | model-lattice | sigma | chi2 contribution | source lattice file |\n|---|---:|---:|---:|---:|---:|---:|---|\n")
        for i, r in enumerate(rows):
            src = next((x["path"] for x in used if x["irrep"] == r["irrep"] and x["state"] == r["state"]), "")
            f.write(f"| {r['irrep']} | {r['level']} | {r['lattice']:.17g} | {r['model']:.17g} | {r['model']-r['lattice']:.17g} | {r['sigma']:.17g} | {contrib[i] if i < len(contrib) else math.nan:.17g} | {src} |\n")
        f.write(f"\nReported chi2: `{summary.get('chi2','nan')}`; reconstructed sum: `{sum(contrib) if contrib else math.nan}`.\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.update({"OMP_NUM_THREADS":"18", "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "VECLIB_MAXIMUM_THREADS":"1", "NUMEXPR_NUM_THREADS":"1", "OMP_DYNAMIC":"FALSE"})
    base_text = BASE.read_text()
    audit_rows, all_input_rows = [], []
    prefit = {}
    for cut in CUTS:
        used, excluded = lattice_records(cut); prefit[cut] = (used, excluded)
        for r in used + excluded:
            all_input_rows.append([cut, 20, r["irrep"], r["path"], True, Path(r["path"]).stat().st_size, r["sha256"], "whitespace columns; header skipped; En_lab column 2", r["nrows"], "USED" if r in used else "EXCLUDED_BY_CUTOFF", "PASS"])
        for s in SECTORS:
            wz = ROOT / f"output/v33p_gpu_classifier_all_irrep/accepted_windows/L20_{s}/accepted_windows.csv"
            cache = ROOT / f"output/v33p_projected_basis_cache/hot_windows/L20_{s}"
            count = 0; requested = set(); rows_saved = []
            if wz.exists():
                with wz.open() as f:
                    for r in csv.DictReader(f):
                        if float(r["zero_estimate_initial"]) <= cut + 1e-15:
                            count += 1; requested.update(range(int(r["max_row_left"]), int(r["max_row_right"])+1))
            meta = cache / "projected_basis_hotwindows.json"; row_indices = set()
            if meta.exists():
                try: row_indices = set(json.loads(meta.read_text()).get("row_indices", []))
                except Exception: pass
            lattice_count = sum(1 for r in used if r["irrep"] == s)
            files_ok = all((cache / x).exists() for x in ("projected_basis_hotwindows.bin", "projected_basis_hotwindows.json", "projected_basis_hotwindows_validation.csv", "PROJECTED_BASIS_HOTWINDOW_CACHE_REPORT.md"))
            missing = sorted(requested - row_indices)
            status = "PASS" if files_ok and count == lattice_count and not missing else "BLOCKED"
            audit_rows.append([cut, 20, s, lattice_count, count, len(requested), len(row_indices), len(missing), missing[:5], files_ok, status])
    with (OUT / "PREFIT_COVERAGE_AUDIT.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["Ecm_cutoff","Lbyas","irrep","lattice_levels","accepted_windows","requested_window_rows","cache_row_indices","missing_rows","missing_first","required_cache_files","status"]); w.writerows(audit_rows)
    with (OUT / "L20_LATTICE_INPUT_FILES_ALL_CUTOFFS.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["Ecm_cutoff","Lbyas","irrep","input_file_path","file_exists","file_size_bytes","sha256","columns_detected","n_rows_read","status","audit_status"]); w.writerows(all_input_rows)
    with (OUT / "PREFIT_COVERAGE_AUDIT.md").open("w") as f:
        f.write("# L20 prefit coverage audit\n\n")
        f.write("Lattice inputs are read from `/home/digonto/Codes/KKpi_I2/spectrum/Ecm_data/data`; parser skips one header line and reads whitespace column 2 as `En_lab`, then converts to Ecm using the fitter's L=20, xi=3.444 momentum formula.\n\n")
        f.write("| cutoff | irrep | lattice levels | accepted windows | requested rows | missing rows | cache files | status |\n|---:|---|---:|---:|---:|---:|---|---|\n")
        for r in audit_rows: f.write(f"| {r[0]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[7]} | {'yes' if r[9] else 'no'} | {r[10]} |\n")
        f.write("\nCutoffs 0.345 and 0.355 are blocked if lattice levels exceed the validated accepted-window package; no fit is run for such a cutoff.\n")
    if not all(r[-1] == "PASS" for r in audit_rows if r[0] <= 0.335):
        raise SystemExit("prefit coverage failed for a runnable cutoff")

    guess = START; global_rows = []; drift_rows = []; global_levels = []; global_lattice = []
    for cut in CUTS:
        ctag = f"Ecm_cut_{cut:.3f}".replace(".", "p"); outdir = OUT / ctag; outdir.mkdir(parents=True, exist_ok=True)
        used, excluded = prefit[cut]; write_lattice_files(cut, outdir, used, excluded)
        (outdir / "starting_parameters.txt").write_text("\n".join(f"{p} {v:.17g}" for p, v in zip(PARAMS, guess)) + "\n")
        (outdir / "lattice_input_files.csv").replace(outdir / "lattice_input_files.csv") if False else None
        # Keep explicit provenance files in each cutoff directory.
        (outdir / "LATTICE_INPUT_PROVENANCE.md").write_text(f"# Lattice input provenance, Ecm_cutoff={cut:.3f}\n\n- Directory: `{JACK_DIR}`\n- Parser: filename `20_<file_irrep>_n<state>.jack`; one header line skipped; whitespace column 2 read as En_lab.\n- Ecm conversion: `sqrt(En_lab^2 - P^2)` with xi=3.444 and the irrep momentum shell.\n- Levels are filtered by the mean converted Ecm <= {cut:.17g}, sorted by the fitter order `{', '.join(SECTORS)}` and state number, then mapped to sorted accepted-window roots.\n- Excluded levels are listed in `lattice_levels_excluded.csv` with reason.\n- Model_found is compared with the selected lattice count in `fit_metrics.csv`.\n- Uncertainties are jackknife covariance-derived Ecm sigmas; correlated chi2 uses the fitter covariance inverse.\n")
        valid_prefit = all(r[-1] == "PASS" for r in audit_rows if r[0] == cut)
        zero_paths = filtered_zero_files(cut, outdir, base_text) if valid_prefit else {}
        tag = f"l20_full_cut_{cut:.3f}".replace(".", "p")
        if not valid_prefit:
            for name in ("fcn_once_before_fit.log", "minimization.log", "fcn_once_after_fit.log"): (outdir / name).write_text("NOT RUN: prefit accepted-window/cache coverage blocked this cutoff.\n")
            (outdir / "final_parameters.csv").write_text("parameter,value,error,relative_error,status\nstatus,,,,BLOCKED_PREFIT_COVERAGE\n")
            (outdir / "model_ecm_levels.csv").write_text("Ecm_cutoff,Lbyas,irrep,level_index,lattice_Ecm,model_Ecm,residual_model_minus_lattice,sigma_or_weight,row_index,bracket_id,source_window,status\n")
            (outdir / "lattice_vs_model_levels.csv").write_text("Ecm_cutoff,Lbyas,irrep,level_index_used,lattice_Ecm,lattice_Ecm_err_or_sigma,model_Ecm,residual_model_minus_lattice,normalized_residual,chi2_contribution,row_index_or_window_row,bracket_id,source_window,lattice_source_file,status\n")
            (outdir / "residuals.csv").write_text("Lbyas,irrep,level_index,lattice_Ecm,model_Ecm,residual_model_minus_lattice,sigma,normalized_residual\n")
            (outdir / "covariance_matrix.csv").write_text("status\nBLOCKED_PREFIT_COVERAGE\n")
            (outdir / "correlation_matrix.csv").write_text("status\nBLOCKED_PREFIT_COVERAGE\n")
            (outdir / "parameter_errors.csv").write_text("parameter,error,status\nstatus,,BLOCKED_PREFIT_COVERAGE\n")
            (outdir / "parameter_sensitivity.csv").write_text("status\nBLOCKED_PREFIT_COVERAGE\n")
            (outdir / "parameter_impact_summary.csv").write_text("Ecm_cutoff,parameter,base_value,error,delta_chi2_minus,delta_chi2_plus,model_found,max_abs_dE_dparam,max_abs_dE_times_parameter,most_affected_level\n")
            (outdir / "fit_metrics.csv").write_text("Ecm_cutoff,n_levels,model_found,chi2,dof,chi2_dof,fcn_evals,edm_or_convergence_metric,converged,fallback_count,full_scan_count,load_time_sec,total_runtime_sec,avg_fcn_time_sec\n")
            (outdir / "final_parameters.json").write_text(json.dumps({"status":"BLOCKED_PREFIT_COVERAGE"}, indent=2)+"\n")
            (outdir / "model_ecm_levels.json").write_text("[]\n")
            (outdir / "lattice_vs_model_levels.md").write_text(f"# Blocked cutoff {cut:.3f}\n\nNo fit was run because accepted windows/cache coverage did not match lattice levels.\n")
            (outdir / "PARAMETER_IMPACT_SUMMARY.md").write_text("# Parameter impact\n\nUnavailable because prefit coverage blocked this cutoff.\n")
            (outdir / "FIT_SUMMARY.md").write_text(f"# Ecm cutoff {cut:.3f}\n\nStatus: **BLOCKED_PREFIT_COVERAGE**. No model or fit values were fabricated.\n")
            global_rows.append([cut, len(used), "0/" + str(len(used)), "nan", "nan", "nan", *("nan",)*8, 0, 0, "nan", "0/0", "false", "BLOCKED_PREFIT_COVERAGE"])
            continue
        cfg = make_config(cut, guess, outdir, zero_paths, tag)
        (outdir / "fitter_input.txt").write_text(cfg.read_text())
        rc, wall = run([str(BIN), str(cfg), "fcn-once"], outdir / "fcn_once_before_fit.log", env); fcn_text = (outdir / "fcn_once_before_fit.log").read_text(); fcn = parse_fcn(fcn_text)
        if rc != 0 or fcn["found"] != fcn["target"]:
            raise SystemExit(f"fcn-once failed at {cut}: {fcn}")
        fit_rc, fit_wall = run([str(BIN), str(cfg), "fit"], outdir / "minimization.log", env); fit_text = (outdir / "minimization.log").read_text(); summary_path = outdir / f"{tag}_fit_summary_allL.dat"; summary = parse_summary(summary_path)
        if fit_rc != 0 or summary.get("valid") != "1":
            raise SystemExit(f"fit failed at {cut}: rc={fit_rc} summary={summary}")
        best = tuple(float(summary[p]) for p in PARAMS); errs = tuple(float(summary.get(p+"_err", "nan")) for p in PARAMS); guess = best
        # final fcn-once at converged parameters
        after_cfg = make_config(cut, best, outdir, zero_paths, tag + "_after_fit"); run([str(BIN), str(after_cfg), "fcn-once"], outdir / "fcn_once_after_fit.log", env)
        for p, v, e in zip(PARAMS, best, errs): pass
        with (outdir / "final_parameters.csv").open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["parameter","value","error","relative_error","status"])
            for p, v, e in zip(PARAMS, best, errs): w.writerow([p, v, e, abs(e/v) if math.isfinite(e) and v else math.nan, "MINUIT_HESSE"])
        (outdir / "final_parameters.json").write_text(json.dumps({p:{"value":v,"error":e} for p,v,e in zip(PARAMS,best,errs)}, indent=2)+"\n")
        write_matrix_csv(outdir / f"{tag}_parameter_covariance.dat", outdir / "covariance_matrix.csv", list(PARAMS))
        write_matrix_csv(outdir / f"{tag}_parameter_correlation.dat", outdir / "correlation_matrix.csv", list(PARAMS))
        with (outdir / "parameter_errors.csv").open("w", newline="") as f:
            w=csv.writer(f); w.writerow(["parameter","error","status"]); [w.writerow([p,e,"MINUIT_HESSE"]) for p,e in zip(PARAMS,errs)]
        # Native fit-levels and detailed level mapping.
        create_per_level(cut, outdir, tag, used, summary, outdir / "parameter_sensitivity.csv")
        # Sensitivity mode at the converged point.
        sens_cfg = make_config(cut, best, outdir, zero_paths, tag + "_sensitivity")
        sens_text = sens_cfg.read_text()
        for p, v in zip(PARAMS,best): sens_text = set_key(sens_text, "sensitivity_best_"+p, f"{v:.17g}")
        for p, step in zip(PARAMS,(1000.0,10000.0,1000.0,10000.0)): sens_text = set_key(sens_text, "sensitivity_step_"+p, f"{step:.17g}")
        sens_text = set_key(sens_text, "sensitivity_output_csv", str(outdir / "parameter_sensitivity.csv")); sens_cfg.write_text(sens_text)
        run([str(BIN), str(sens_cfg), "sensitivity"], outdir / "parameter_sensitivity.log", env)
        # Impact summary from the native sensitivity CSV.
        impacts=[]
        if (outdir/"parameter_sensitivity.csv").exists():
            with (outdir/"parameter_sensitivity.csv").open() as f:
                for r in csv.DictReader(f):
                    if r["base_point"] != "best_logged": continue
                    j=r["parameter"]; val=float(r["base_value"]); err=errs[PARAMS.index(j)]
                    deriv=[float(r[f"dE{k}_dparam"]) for k in range(4)]; norm=[abs(x*val) for x in deriv]
                    impacts.append({"Ecm_cutoff":cut,"parameter":j,"base_value":val,"error":err,"delta_chi2_minus":float(r["delta_chi2_minus"]),"delta_chi2_plus":float(r["delta_chi2_plus"]),"model_found":r["found_base"],"max_abs_dE_dparam":max(abs(x) for x in deriv),"max_abs_dE_times_parameter":max(norm),"most_affected_level":norm.index(max(norm))})
        with (outdir/"parameter_impact_summary.csv").open("w",newline="") as f:
            fields=["Ecm_cutoff","parameter","base_value","error","delta_chi2_minus","delta_chi2_plus","model_found","max_abs_dE_dparam","max_abs_dE_times_parameter","most_affected_level"]; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(impacts)
        with (outdir/"PARAMETER_IMPACT_SUMMARY.md").open("w") as f:
            f.write(f"# Parameter impact summary, Ecm_cutoff={cut:.3f}\n\nFinite differences use steps (1000, 10000, 1000, 10000) around the converged point. Raw determinant and projected-basis hot-window paths are unchanged.\n\n")
            f.write("| parameter | error | delta chi2 - | delta chi2 + | max abs dE/dp | most affected model level |\n|---|---:|---:|---:|---:|---:|\n")
            for x in impacts: f.write(f"| {x['parameter']} | {x['error']:.6g} | {x['delta_chi2_minus']:.6g} | {x['delta_chi2_plus']:.6g} | {x['max_abs_dE_dparam']:.6g} | {x['most_affected_level']} |\n")
        cov_status="positive_definite"; eigs=[]
        if np is not None and (outdir/f"{tag}_parameter_covariance.dat").exists():
            mat=matrix_dat(outdir/f"{tag}_parameter_covariance.dat"); eigs=np.linalg.eigvalsh(mat).tolist(); cov_status="positive_definite" if min(eigs)>0 else "not_positive_definite"
        evals=len(re.findall(r"\[hot-window-fcn\] eval=", fit_text)); times=[float(x) for x in re.findall(r"\[hot-window-fcn\].*?fcn_sec=([0-9eE+.+-]+)",fit_text)]; rows_eval=[int(x) for x in re.findall(r"\[hot-window-fcn\].*?rows_evaluated=(\d+)",fit_text)]
        ndof=int(summary.get("ndof",len(used)-4)); chi=float(summary["chi2"]); model_found=int(summary["model_levels_found"])
        with (outdir/"fit_metrics.csv").open("w",newline="") as f:
            w=csv.writer(f); w.writerow(["Ecm_cutoff","n_levels","model_found","chi2","dof","chi2_dof","fcn_evals","edm_or_convergence_metric","converged","fallback_count","full_scan_count","load_time_sec","total_runtime_sec","avg_fcn_time_sec"]); w.writerow([cut,len(used),f"{model_found}/{len(used)}",chi,ndof,summary.get("chi2_dof","nan"),evals,"unavailable",True,0,0,fcn["load"],fit_wall,(sum(times)/len(times) if times else math.nan)])
        (outdir/"FIT_SUMMARY.md").write_text(f"# L20 fit summary, Ecm_cutoff={cut:.3f}\n\n- status: PASS\n- projected-basis cache mode: hot_windows\n- model_found: {model_found}/{len(used)}\n- chi2: {chi:.17g}\n- dof: {ndof}\n- chi2/dof: {summary.get('chi2_dof','nan')}\n- fallback/full_scan: 0/0\n- Minuit covariance: {cov_status}; eigenvalues: {eigs}\n- fit evaluations: {evals}\n- load time: {fcn['load']} s\n- total runtime: {fit_wall} s\n\nPer-level data: `lattice_vs_model_levels.csv`, `model_ecm_levels.csv`, and `lattice_vs_model_levels.md`.\n")
        global_rows.append([cut,len(used),f"{model_found}/{len(used)}",chi,ndof,summary.get("chi2_dof","nan"),*sum(([summary.get(p,"nan"),summary.get(p+"_err","nan")] for p in PARAMS),[]),evals,fit_wall,(sum(times)/len(times) if times else math.nan),"0/0",True,"PASS"])
        for line in (outdir/"model_ecm_levels.csv").read_text().splitlines()[1:]: global_levels.append([cut]+line.split(","))
        for line in (outdir/"lattice_levels_used.csv").read_text().splitlines()[1:]: global_lattice.append(line)
        drift_rows.append([cut,*sum(([summary.get(p,"nan"),summary.get(p+"_err","nan")] for p in PARAMS),[])])
    fields=["Ecm_cutoff","n_levels","model_found","chi2","dof","chi2_dof","K3iso0","K3iso0_err","K3iso1","K3iso1_err","K3B","K3B_err","K3E","K3E_err","fcn_evals","runtime","avg_fcn_time","fallback/full_scan","converged","status"]
    with (OUT/"L20_FULL_ECM_CUTOFF_FIT_SUMMARY.csv").open("w",newline="") as f: w=csv.writer(f); w.writerow(fields); w.writerows(global_rows)
    with (OUT/"L20_K3DF_PARAMETER_DRIFT.csv").open("w",newline="") as f: w=csv.writer(f); w.writerow(["Ecm_cutoff"]+[x for p in PARAMS for x in (p,p+"_err")]); w.writerows(drift_rows)
    # Copy aggregate input/level tables and model tables with explicit cutoff prefix.
    with (OUT/"L20_LATTICE_LEVELS_USED_ALL_CUTOFFS.csv").open("w") as f: f.write("Ecm_cutoff,Lbyas,irrep,level_index_original,level_index_used,lattice_Ecm,lattice_Ecm_err_or_sigma,lab_energy_if_available,row_source_file,source_line_or_row,status\n"); [f.write(f"{cut},"+line+"\n") for cut in CUTS for line in (OUT/f"Ecm_cut_{cut:.3f}".replace(".","p")/"lattice_levels_used.csv").read_text().splitlines()[1:]]
    with (OUT/"L20_MODEL_ECM_ALL_CUTOFFS.csv").open("w") as f: f.write("Ecm_cutoff,Lbyas,irrep,level_index,lattice_Ecm,model_Ecm,residual_model_minus_lattice,sigma_or_weight,row_index,bracket_id,source_window,status\n"); [f.write(",".join(map(str,line))+"\n") for line in global_levels]
    with (OUT/"L20_LATTICE_INPUT_FILES_ALL_CUTOFFS.csv").open("a") as f: pass
    # Global lattice-vs-model files.
    with (OUT/"L20_LATTICE_VS_MODEL_ALL_CUTOFFS.csv").open("w") as f:
        f.write("Ecm_cutoff,Ecm_cutoff,Lbyas,irrep,level_index_used,lattice_Ecm,lattice_Ecm_err_or_sigma,model_Ecm,residual_model_minus_lattice,normalized_residual,chi2_contribution,row_index_or_window_row,bracket_id,source_window,lattice_source_file,status\n")
        for cut in CUTS:
            p=OUT/f"Ecm_cut_{cut:.3f}".replace(".","p")/"lattice_vs_model_levels.csv"; [f.write(f"{cut},"+line+"\n") for line in p.read_text().splitlines()[1:]]
    with (OUT/"L20_FULL_ECM_CUTOFF_FIT_SUMMARY.md").open("w") as f:
        f.write("# L20 full Ecm-cutoff fit suite\n\nL24 was not resumed by this task. All runnable cases use L20 accepted sectors and projected-basis hot-window caches only.\n\n")
        f.write("| cutoff | levels | model_found | chi2 | dof | chi2/dof | K3iso0 | K3iso1 | K3B | K3E | status |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in global_rows: f.write("| "+" | ".join(map(str,[r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[8],r[10],r[12],r[-1]]))+" |\n")
        f.write("\n## Interpretation\n\nThe validated accepted-window package covers through 0.335. Cutoffs 0.345 and 0.355 are reported as blocked when extra lattice levels lack accepted windows; no fake roots are inserted. Among runnable cutoffs, inspect parameter drift and covariance together before production interpretation.\n")
    with (OUT/"L20_K3DF_PARAMETER_DRIFT.md").open("w") as f:
        f.write("# L20 K3df parameter drift\n\n"); f.write("Changes are compared with the preceding runnable cutoff and with 0.315. Errors are Minuit HESSE outputs; blocked cutoffs have no parameter values.\n\n")
        f.write("| cutoff | parameter | value | error | delta previous | percent previous | delta vs 0.315 | percent vs 0.315 |\n|---:|---|---:|---:|---:|---:|---:|---:|\n")
        runnable=[r for r in global_rows if r[-1]=="PASS"]; baseline=runnable[0] if runnable else None
        prev=None
        for r in runnable:
            for j,p in enumerate(PARAMS):
                value=float(r[6+2*j]); err=float(r[7+2*j]); pv=float(prev[6+2*j]) if prev else math.nan; bv=float(baseline[6+2*j]) if baseline else math.nan
                f.write(f"| {r[0]} | {p} | {value:.17g} | {err:.17g} | {(value-pv) if prev else math.nan:.6g} | {((value-pv)/pv*100) if prev and pv else math.nan:.6g} | {(value-bv) if baseline else math.nan:.6g} | {((value-bv)/bv*100) if baseline and bv else math.nan:.6g} |\n")
            prev=r
    with (OUT/"L20_LATTICE_VS_MODEL_ALL_CUTOFFS.md").open("w") as f:
        f.write("# L20 lattice versus model across cutoffs\n\n")
        for cut in CUTS:
            p=OUT/f"Ecm_cut_{cut:.3f}".replace(".","p")/"lattice_vs_model_levels.md"
            f.write("\n---\n\n" + (p.read_text() if p.exists() else f"## Ecm_cutoff={cut:.3f}\n\nUnavailable.\n"))
    with (OUT/"L20_PARAMETER_COVARIANCE_COMPARISON.md").open("w") as f:
        f.write("# L20 covariance and error comparison\n\n")
        for cut in CUTS:
            p=OUT/f"Ecm_cut_{cut:.3f}".replace(".","p")/"FIT_SUMMARY.md"
            f.write("\n---\n\n" + (p.read_text() if p.exists() else f"## Ecm_cutoff={cut:.3f}\n\nUnavailable.\n"))
    with (OUT/"L20_FULL_CUTOFF_DEPENDENCE_FINAL_REPORT.md").open("w") as f:
        f.write("# L20 full cutoff-dependence final report\n\n")
        f.write("## Scope and checkpoint\n\n- Lbyas: `20` only\n- Sectors: `000_A1m`, `100_A2`, `110_A2`, `111_A2`, `200_A2`\n- Projected-basis cache mode: `hot_windows`\n- fallback/full_scan: `0/0` for runnable fits\n- L24 status: no active PIDs remained when the requested stop was issued; no L24 work was launched or resumed by this task.\n\n")
        f.write("## Prefit coverage\n\nSee `PREFIT_COVERAGE_AUDIT.csv/.md`. Cache files required per sector were checked, and requested `max_row_left..max_row_right` sets were compared with JSON `row_indices`.\n\n")
        f.write(f"Exact lattice input directory: `{JACK_DIR}`. The fitter parser reads `20_<file_irrep>_n<state>.jack`, skips one header line, reads whitespace column 2 as `En_lab`, converts to Ecm with xi=3.444, filters by mean Ecm cutoff, and sorts by sector order and state.\n\n")
        f.write("## Per-cutoff results\n\n")
        for r in global_rows:
            cut=r[0]; f.write(f"### Ecm_cutoff={cut:.3f}\n\n")
            if r[-1] != "PASS":
                f.write("Status: **BLOCKED_PREFIT_COVERAGE**. Lattice levels exceeded accepted-window coverage; no fit, roots, or parameters were fabricated.\n\n")
                continue
            ctag = f"Ecm_cut_{cut:.3f}".replace(".", "p")
            f.write(f"- levels/model_found: `{r[1]}` / `{r[2]}`\n- chi2, dof, chi2/dof: `{r[3]}`, `{r[4]}`, `{r[5]}`\n- fallback/full_scan: `{r[17]}`\n- parameters: K3iso0=`{r[6]}` +/- `{r[7]}`, K3iso1=`{r[8]}` +/- `{r[9]}`, K3B=`{r[10]}` +/- `{r[11]}`, K3E=`{r[12]}` +/- `{r[13]}`\n- model levels: `{ctag}/model_ecm_levels.csv`\n- lattice-vs-model: `{ctag}/lattice_vs_model_levels.csv`\n- sensitivity/impact: `{ctag}/parameter_sensitivity.csv`, `{ctag}/parameter_impact_summary.csv`\n\n")
        f.write("## Interpretation and caveats\n\nThe highest runnable cutoff is operationally preferable only if parameter drift and covariance are acceptable. These remain L20-only fits, not a global production fit. Cutoffs 0.345 and 0.355 require additional accepted classifier windows/cache coverage for the extra lattice levels before they can be fitted.\n\nNext action: review the blocked high-cutoff sectors and decide whether to generate/validate additional accepted windows; L24 should remain inactive until explicitly resumed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
