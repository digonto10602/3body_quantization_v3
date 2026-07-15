#!/usr/bin/env python3
"""Run the bounded L20 projected-basis cutoff suite sequentially."""
from __future__ import annotations

import csv
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/v33p_projected_basis_cache/fitter_projected_basis_hot_windows_all.in"
OUT = ROOT / "output/v33p_projected_basis_cache/l20_cutoff_fits"
BIN = ROOT / "bin/v33f_k3df_fitter_multiL_v33e"
CUTS = (0.305, 0.315, 0.325, 0.335)
PARAMS = ("K3iso0", "K3iso1", "K3B", "K3E")
START = (73735.840894011912, -972421.14060757787, 347174.05548116949, -1226756.7068845264)


def set_key(text: str, key: str, value: str) -> str:
    pat = re.compile(rf"(?m)^\s*{re.escape(key)}\s*=.*$")
    line = f"{key} = {value}"
    return pat.sub(line, text, count=1) if pat.search(text) else text + "\n" + line + "\n"


def parse_summary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        bits = line.split()
        if len(bits) >= 2:
            out[bits[0]] = bits[1]
            if bits[0] in PARAMS and len(bits) >= 4:
                out[bits[0] + "_err"] = bits[3]
    return out


def last_float(pattern: str, text: str, default: float = math.nan) -> float:
    vals = re.findall(pattern, text)
    return float(vals[-1]) if vals else default


def last_int(pattern: str, text: str, default: int = 0, group: int = 1) -> int:
    vals = re.findall(pattern, text)
    if not vals:
        return default
    value = vals[-1]
    if isinstance(value, tuple):
        value = value[group - 1]
    return int(value)


def make_config(cut: float, guess: tuple[float, ...], outdir: Path) -> Path:
    text = BASE.read_text()
    tag = f"l20_ecmcut_{cut:.3f}".replace(".", "p")
    replacements = {
        "Lbyas_values": "20",
        "irreps_L24": "",
        "list_of_mom": "000_A1m 100_A2 110_A2 111_A2 200_A2",
        "Ecm_cutoff": f"{cut:.17g}",
        "energy_cutoff": f"{cut:.17g}",
        "one_fcn_only": "0",
        "max_fcn": "200",
        "max_fcn_evals": "250",
        "output_dir": str(outdir),
        "output_tag": tag,
        "K3iso0_guess": f"{guess[0]:.17g}",
        "K3iso1_guess": f"{guess[1]:.17g}",
        "K3B_guess": f"{guess[2]:.17g}",
        "K3E_guess": f"{guess[3]:.17g}",
        "K3iso0_step": "1000.0",
        "K3iso1_step": "10000.0",
        "K3B_step": "1000.0",
        "K3E_step": "10000.0",
        "float_K3iso0": "1",
        "float_K3iso1": "1",
        "float_K3B": "1",
        "float_K3E": "1",
    }
    for key, value in replacements.items():
        text = set_key(text, key, value)
    # The native loader honors the CSV's inside_Ecm_cutoff flag.  Make a
    # cutoff-specific accepted-zero view so each fit uses its own cutoff
    # without changing fitter/classifier logic.
    for match in re.finditer(r"(?m)^\s*(accepted_zeros_file_L20_[^ ]+)\s*=\s*(\S+)\s*$", text):
        key, raw_path = match.group(1), match.group(2)
        src = Path(raw_path)
        if not src.is_absolute():
            src = ROOT / src
        if not src.exists():
            continue
        with src.open(newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            kept = []
            for row in reader:
                try:
                    is_true = row.get("user_label", "").strip().lower() == "true"
                    z = float(row.get("zero_estimate", "nan"))
                except ValueError:
                    continue
                if is_true and math.isfinite(z) and z <= cut + 1e-15:
                    if "inside_Ecm_cutoff" in row:
                        row["inside_Ecm_cutoff"] = "true"
                    kept.append(row)
        filtered = outdir / (Path(raw_path).stem + f"_cut{cut:.3f}".replace(".", "p") + ".csv")
        with filtered.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(kept)
        text = set_key(text, key, str(filtered))
    path = outdir / "fitter_input.in"
    path.write_text(text)
    return path


def run_logged(cmd: list[str], log: Path, env: dict[str, str]) -> tuple[int, float]:
    start = time.monotonic()
    with log.open("w") as f:
        f.write("COMMAND: " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode, time.monotonic() - start


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "18", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE",
    })
    guess = START
    rows: list[dict[str, object]] = []
    for cut in CUTS:
        ctag = f"Ecm_cut_{cut:.3f}".replace(".", "p")
        outdir = OUT / ctag
        outdir.mkdir(parents=True, exist_ok=True)
        start_guess = guess
        cfg = make_config(cut, start_guess, outdir)
        (outdir / "run_environment.txt").write_text("\n".join(f"{k}={env[k]}" for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS", "OMP_DYNAMIC")) + "\n")
        fcn_log = outdir / "fcn_once.log"
        rc_fcn, fcn_wall = run_logged([str(BIN), str(cfg), "fcn-once"], fcn_log, env)
        fcn_text = fcn_log.read_text(errors="replace")
        fcn_found = last_int(r"model_found=(\d+)/(\d+)", fcn_text, group=1)
        fcn_target = int(re.findall(r"model_found=(\d+)/(\d+)", fcn_text)[-1][1]) if re.findall(r"model_found=(\d+)/(\d+)", fcn_text) else 0
        fcn_chi = last_float(r"\[fcn-once\] chi2=([0-9eE+\-.]+)", fcn_text)
        fcn_rows = last_int(r"rows_evaluated=(\d+)", fcn_text)
        fcn_sec = last_float(r"fcn_sec=([0-9eE+\-.]+)", fcn_text)
        fcn_load = last_float(r"cache_load_and_precompute_sec=([0-9eE+\-.]+)", fcn_text)
        fit_rc = -1
        fit_wall = 0.0
        summary: dict[str, str] = {}
        fit_text = ""
        if rc_fcn == 0 and fcn_found == fcn_target and fcn_target > 0:
            fit_rc, fit_wall = run_logged([str(BIN), str(cfg), "fit"], outdir / "minimization.log", env)
            fit_text = (outdir / "minimization.log").read_text(errors="replace")
            # The native output tag is l20_ecmcut_0pXXX.
            tag = f"l20_ecmcut_{cut:.3f}".replace(".", "p")
            native_summary = outdir / f"{tag}_fit_summary_allL.dat"
            summary = parse_summary(native_summary)
            lev = outdir / f"{tag}_fit_levels_allL.dat"
            if lev.exists():
                shutil.copyfile(lev, outdir / "roots_model_found.dat")
                with lev.open() as src, (outdir / "roots_model_found.csv").open("w", newline="") as dst:
                    lines = [x for x in src if not x.startswith("#")]
                    w = csv.writer(dst)
                    w.writerow(["row", "Lbyas", "file_irrep", "internal_irrep", "state", "level_index", "lattice_Ecm", "lattice_err", "model_Ecm", "residual", "shifted_from_lab"])
                    for line in lines:
                        w.writerow(line.split())
            if summary.get("valid") == "1":
                guess = tuple(float(summary[p]) for p in PARAMS)  # type: ignore[assignment]
        final_valid = summary.get("valid", "0")
        fit_found = int(summary.get("model_levels_found", "0")) if summary else 0
        ndata = int(summary.get("ndata", str(fcn_target or 0))) if summary else fcn_target
        npar = int(summary.get("npar", "4")) if summary else 4
        ndof = int(summary.get("ndof", str(ndata - npar))) if summary else ndata - npar
        evals = len(re.findall(r"\[hot-window-fcn\] eval=", fit_text))
        fit_times = [float(x) for x in re.findall(r"\[hot-window-fcn\].*?fcn_sec=([0-9eE+\-.]+)", fit_text)]
        fit_rows = [int(x) for x in re.findall(r"\[hot-window-fcn\].*?rows_evaluated=(\d+)", fit_text)]
        fit_load = last_float(r"cache_load_and_precompute_sec=([0-9eE+\-.]+)", fit_text, fcn_load)
        minuit_valid = last_int(r"\[final\] Minuit valid=(\d+)", fit_text, 0)
        status = "PASS" if fit_rc == 0 and final_valid == "1" and fit_found == ndata else "FAIL_OR_NOT_CONVERGED"
        final_param_values = {p: summary.get(p, "nan") for p in PARAMS}
        final_errors = {p: summary.get(p + "_err", "nan") for p in PARAMS}
        report = [
            f"# L20 cutoff fit: Ecm_cutoff={cut:.3f}", "", "## Status", "",
            f"- status: **{status}**", f"- fcn-once return: {rc_fcn}", f"- fit return: {fit_rc}",
            f"- projected-basis mode: `hot_windows`", f"- fallback/full_scan: `0/0` (required)",
            f"- model_found at fcn-once: `{fcn_found}/{fcn_target}`", f"- model_found final: `{fit_found}/{ndata}`",
            f"- Minuit valid: `{minuit_valid}`", f"- nominal dof: `{ndof}`", "",
            "## Parameters", "", "| parameter | start | final | error |", "|---|---:|---:|---:|",
        ]
        for p, s, e in zip(PARAMS, start_guess, PARAMS):
            report.append(f"| {p} | {s:.17g} | {final_param_values[p]} | {final_errors[p]} |")
        report += ["", "## Timing and diagnostics", "", f"- fcn-once chi2: `{fcn_chi}`", f"- final chi2: `{summary.get('chi2', 'nan')}`", f"- chi2/dof: `{summary.get('chi2_dof', 'nan')}`", f"- fcn-once wall: `{fcn_wall:.9g} s`", f"- fcn-once rows: `{fcn_rows}`", f"- fcn-once FCN: `{fcn_sec:.9g} s`", f"- cache load/precompute: `{fit_load:.9g} s`", f"- fit FCN evaluations: `{evals}`", f"- average fit FCN: `{(sum(fit_times)/len(fit_times)) if fit_times else math.nan:.9g} s`", f"- max fit FCN: `{max(fit_times) if fit_times else math.nan:.9g} s`", f"- max rows per fit FCN: `{max(fit_rows) if fit_rows else 0}`", "", "Native fitter outputs are preserved in this directory."]
        (outdir / "fit_summary.md").write_text("\n".join(report) + "\n")
        (outdir / "final_parameters.dat").write_text("\n".join(f"{p} {final_param_values[p]} err {final_errors[p]}" for p in PARAMS) + "\n")
        rows.append({"Ecm_cutoff": f"{cut:.3f}", "included_sectors": "L20/000_A1m;L20/100_A2;L20/110_A2;L20/111_A2;L20/200_A2", "number_of_levels": ndata, "model_found": f"{fit_found}/{ndata}", "chi2": summary.get("chi2", "nan"), "chi2_dof": summary.get("chi2_dof", "nan"), **final_param_values, **{p + "_err": final_errors[p] for p in PARAMS}, "FCN_evaluations": evals, "total_runtime_s": f"{fit_wall:.9g}", "average_FCN_time_s": f"{(sum(fit_times)/len(fit_times)) if fit_times else math.nan:.9g}", "fallback_full_scan": "0/0", "status": status})
    fields = list(rows[0]) if rows else []
    with (OUT / "L20_ECM_CUTOFF_DEPENDENCE_SUMMARY.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    md = ["# L20 Ecm cutoff dependence", "", "Only L20 accepted sectors were included; all fits used projected-basis hot-window caches, `accepted_windows`, and `fallback_full_scan=0`.", "", "| cutoff | levels | model_found | chi2 | chi2/dof | K3iso0 | K3iso1 | K3B | K3E | FCN evals | status |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        md.append("| {Ecm_cutoff} | {number_of_levels} | {model_found} | {chi2} | {chi2_dof} | {K3iso0} | {K3iso1} | {K3B} | {K3E} | {FCN_evaluations} | {status} |".format(**r))
    md += ["", "## Interpretation", "", "The recommended cutoff is the highest cutoff with complete roots, finite chi2, and stable parameters. This report records nominal dof as levels minus four; non-positive dof is a plumbing result, not a statistically meaningful goodness-of-fit measure.", "", "Sequential warm starts were used only after a preceding fit reported `valid=1`."]
    (OUT / "L20_ECM_CUTOFF_DEPENDENCE_SUMMARY.md").write_text("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
