#!/usr/bin/env python3
"""Visualize one fish IBFE/EB run directory from its diagnostic CSV files."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CSV file map — keys used throughout this script
# ---------------------------------------------------------------------------
CSV_FILES = {
    "midline":   "midline_history.csv",
    "curvature": "curvature_phase_diag.csv",
    "force":     "force_decomposition_diag.csv",
    "direction": "direction_debug.csv",
    "geometry":  "geometry_conservation_diag.csv",
    "section":   "section_moment_decomposition.csv",
}

# Columns that must be numeric; rows with NaN in these are dropped.
REQUIRED_NUMERIC_COLUMNS = {
    "midline":   ["time", "station", "s_norm", "x_lab", "y_lab",
                  "x_cm", "y_cm", "theta_body", "x_body", "y_body",
                  "y_prop", "curvature"],
    "curvature": ["step", "time", "s_norm", "x_cm", "y_cm", "kappa_body"],
    "force":     ["step", "time", "x_cm", "y_cm", "v_forward"],
    "direction": ["step", "time", "x_cm", "y_cm", "v_forward"],
    "geometry":  ["step", "time", "area_rel_error", "J_min", "J_max",
                  "J_rms_error", "J_max_abs_error"],
    "section":   ["step", "time", "bin", "s_norm", "M_sum"],
}

# EB force-decomposition component names (from force_decomp_component_name())
FORCE_COMPONENTS = ["passive", "eb_passive", "damping", "active", "sum"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_csv(case_dir: Path, key: str) -> pd.DataFrame | None:
    path = case_dir / CSV_FILES[key]
    if not path.exists():
        print(f"[skip] missing {path.name}")
        return None
    print(f"[read] {path.name}")
    try:
        df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(path, low_memory=False,
                         error_bad_lines=False, warn_bad_lines=True)
    required = [c for c in REQUIRED_NUMERIC_COLUMNS.get(key, []) if c in df.columns]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if required:
        before = len(df)
        df = df.dropna(subset=required).copy()
        dropped = before - len(df)
        if dropped:
            print(f"[clean] dropped {dropped} incomplete rows from {path.name}")
    return df


def numeric(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def max_abs(series: pd.Series) -> float:
    arr = numeric(series)
    return float(np.max(np.abs(arr))) if arr.size else math.nan


def finite_min(series: pd.Series) -> float:
    arr = numeric(series)
    return float(np.min(arr)) if arr.size else math.nan


def finite_max(series: pd.Series) -> float:
    arr = numeric(series)
    return float(np.max(arr)) if arr.size else math.nan


def fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.6e}"


def dedupe_time_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per time (collapse per-station/bin repeats)."""
    if df is None or "time" not in df.columns:
        return pd.DataFrame()
    return df.sort_values("time").groupby("time", as_index=False).first().sort_values("time")


def nearest_times(times: np.ndarray, requested: list[float]) -> list[float]:
    finite = np.asarray(times[np.isfinite(times)], dtype=float)
    if finite.size == 0:
        return []
    selected: list[float] = []
    for target in requested:
        value = float(finite[np.argmin(np.abs(finite - target))])
        if value not in selected:
            selected.append(value)
    return selected


def symmetric_limits(values: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lim = float(np.nanpercentile(np.abs(finite), percentile))
    if lim <= 0 or not np.isfinite(lim):
        lim = float(np.nanmax(np.abs(finite)))
    if lim <= 0 or not np.isfinite(lim):
        lim = 1.0
    return -lim, lim


def savefig(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] {out_path}")


def _plot_cols(ax, df: pd.DataFrame, cols: list[str], **kw) -> None:
    """Plot whichever of `cols` exist in df, skipping missing ones silently."""
    for col in cols:
        if col in df and numeric(df[col]).size:
            ax.plot(df["time"], df[col], label=col, **kw)


# ---------------------------------------------------------------------------
# Run overview
# ---------------------------------------------------------------------------

def plot_run_overview(
    direction: pd.DataFrame | None,
    force: pd.DataFrame | None,
    geometry: pd.DataFrame | None,
    curvature: pd.DataFrame | None,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    axes = axes.ravel()

    # --- panel 0: COM position and forward speed ---
    src = direction if (direction is not None and not direction.empty) else \
          (dedupe_time_rows(force) if force is not None and not force.empty else None)
    if src is not None and not src.empty:
        _plot_cols(axes[0], src, ["x_cm", "y_cm", "v_forward", "v_fwd_cycle_avg"])
        axes[0].set_title("COM position and forward speed")
        axes[0].set_xlabel("time")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].axis("off")

    # --- panel 1: geometry conservation ---
    if geometry is not None and not geometry.empty:
        _plot_cols(axes[1], geometry,
                   ["area_rel_error", "J_rms_error", "J_max_abs_error"])
        # semilogy requires positive values
        for line in axes[1].get_lines():
            y = line.get_ydata()
            if np.any(y > 0):
                break
        axes[1].set_yscale("log")
        axes[1].set_title("Geometry conservation")
        axes[1].set_xlabel("time")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis("off")

    # --- panel 2: forward force components on fluid (weak form) ---
    if force is not None and not force.empty:
        _plot_cols(axes[2], force, [
            "F_total_forward_on_fluid",
            "F_weak_passive_forward_on_fluid",
            "F_weak_eb_passive_forward_on_fluid",
            "F_weak_damping_forward_on_fluid",
            "F_weak_active_forward_on_fluid",
            "F_weak_sum_forward_on_fluid",
        ])
        axes[2].set_title("Forward force components on fluid")
        axes[2].set_xlabel("time")
        axes[2].legend(fontsize=7)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis("off")

    # --- panel 3: wave/drive indices ---
    scalar = dedupe_time_rows(curvature) if curvature is not None else pd.DataFrame()
    if not scalar.empty:
        _plot_cols(axes[3], scalar,
                   ["traveling_wave_index", "signed_traveling_wave_index", "drive_following_index"])
        axes[3].set_title("Wave/drive indices")
        axes[3].set_xlabel("time")
        axes[3].legend(fontsize=8)
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].axis("off")

    savefig(fig, out_dir / "run_overview.png")


# ---------------------------------------------------------------------------
# Direction-debug (motion) detail
# ---------------------------------------------------------------------------

def plot_direction_debug(direction: pd.DataFrame | None, out_dir: Path) -> None:
    if direction is None or direction.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    _plot_cols(axes[0], direction, ["v_forward", "v_lateral", "v_fwd_cycle_avg"])
    axes[0].set_title("Velocity components")
    axes[0].set_ylabel("speed")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    _plot_cols(axes[1], direction, ["a_forward"])
    axes[1].set_title("Forward acceleration")
    axes[1].set_ylabel("acceleration")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    _plot_cols(axes[2], direction, ["F_CM_fwd", "cm_impulse_fwd",
                                    "F_IB_on_fluid_fwd", "F_IB_impulse_fwd",
                                    "tail_A_norm"])
    axes[2].set_title("Forces and tail amplitude")
    axes[2].set_xlabel("time")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    savefig(fig, out_dir / "direction_debug.png")


# ---------------------------------------------------------------------------
# Force / power decomposition
# ---------------------------------------------------------------------------

def plot_force_power(force: pd.DataFrame | None, out_dir: Path) -> None:
    if force is None or force.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Instantaneous power: EB-version components
    _plot_cols(axes[0], force, [
        "P_IB_on_fluid",
        "P_passive", "P_eb_passive", "P_damping", "P_active", "P_sum",
        "P_abs_sum",
    ])
    axes[0].set_title("Power diagnostics (EB version)")
    axes[0].set_ylabel("power")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Accumulated work
    _plot_cols(axes[1], force, [
        "W_IB_on_fluid",
        "W_passive", "W_eb_passive", "W_damping", "W_active", "W_sum",
    ])
    axes[1].set_title("Accumulated work")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("work")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    savefig(fig, out_dir / "force_power.png")


# ---------------------------------------------------------------------------
# Midline
# ---------------------------------------------------------------------------

def plot_midline_snapshots(midline: pd.DataFrame | None, out_dir: Path) -> None:
    if midline is None or midline.empty:
        return
    times = np.sort(midline["time"].dropna().unique())
    sample_times = nearest_times(
        times, [times[0], 0.25 * times[-1], 0.5 * times[-1], 0.75 * times[-1], times[-1]]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for t in sample_times:
        frame = midline[np.isclose(midline["time"], t)].sort_values("s_norm")
        axes[0].plot(frame["x_lab"], frame["y_lab"], label=f"t={t:.3g}")
        axes[1].plot(frame["s_norm"], frame["curvature"], label=f"t={t:.3g}")
    axes[0].set_title("Midline snapshots (lab frame)")
    axes[0].set_xlabel("x_lab")
    axes[0].set_ylabel("y_lab")
    axes[0].axis("equal")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Curvature snapshots")
    axes[1].set_xlabel("s_norm")
    axes[1].set_ylabel("curvature")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    savefig(fig, out_dir / "midline_snapshots.png")


def plot_midline_heatmaps(midline: pd.DataFrame | None, out_dir: Path) -> None:
    if midline is None or midline.empty:
        return
    table_k = midline.pivot_table(index="time", columns="s_norm",
                                   values="curvature", aggfunc="mean").sort_index()
    table_y = midline.pivot_table(index="time", columns="s_norm",
                                   values="y_prop", aggfunc="mean").sort_index()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    x = table_k.columns.to_numpy(dtype=float)
    y = table_k.index.to_numpy(dtype=float)
    vmin, vmax = symmetric_limits(table_k.to_numpy())
    im0 = axes[0].imshow(table_k.to_numpy(), aspect="auto", origin="lower",
                         extent=[x.min(), x.max(), y.min(), y.max()],
                         cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[0].set_title("Curvature κ(s,t)")
    axes[0].set_ylabel("time")
    fig.colorbar(im0, ax=axes[0], label="curvature")

    x2 = table_y.columns.to_numpy(dtype=float)
    y2 = table_y.index.to_numpy(dtype=float)
    vmin, vmax = symmetric_limits(table_y.to_numpy())
    im1 = axes[1].imshow(table_y.to_numpy(), aspect="auto", origin="lower",
                         extent=[x2.min(), x2.max(), y2.min(), y2.max()],
                         cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[1].set_title("Propulsive displacement y_prop(s,t)")
    axes[1].set_xlabel("s_norm")
    axes[1].set_ylabel("time")
    fig.colorbar(im1, ax=axes[1], label="y_prop")
    savefig(fig, out_dir / "midline_heatmaps.png")


# ---------------------------------------------------------------------------
# Curvature phase diagnostics
# ---------------------------------------------------------------------------

def plot_curvature_phase(curvature: pd.DataFrame | None, out_dir: Path) -> None:
    if curvature is None or curvature.empty:
        return
    scalar = dedupe_time_rows(curvature)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.ravel()

    _plot_cols(axes[0], scalar, ["x_cm", "y_cm", "y_cm_relative"])
    axes[0].set_title("COM motion")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    _plot_cols(axes[1], scalar, ["curvature_phase", "curvature_phase_unwrapped", "phase_lag"])
    axes[1].set_title("Phase diagnostics")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    _plot_cols(axes[2], scalar,
               ["curvature_phase_slope", "active_phase_slope_abs", "active_phase_slope_expected"])
    axes[2].set_title("Phase slope")
    axes[2].set_xlabel("time")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    _plot_cols(axes[3], scalar,
               ["active_power", "effective_active_power",
                "active_signed_work", "effective_active_signed_work"])
    axes[3].set_title("Active power / work")
    axes[3].set_xlabel("time")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    savefig(fig, out_dir / "curvature_phase_diagnostics.png")

    # Heatmaps
    table_k = curvature.pivot_table(index="time", columns="s_norm",
                                     values="kappa_body", aggfunc="mean").sort_index()
    table_d = curvature.pivot_table(index="time", columns="s_norm",
                                     values="activation_drive", aggfunc="mean").sort_index()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for table, ax, title, label in [
        (table_k, axes[0], "Body-frame curvature κ(s,t)", "kappa_body"),
        (table_d, axes[1], "Activation drive template(s,t)", "activation_drive"),
    ]:
        x = table.columns.to_numpy(dtype=float)
        y = table.index.to_numpy(dtype=float)
        vmin, vmax = symmetric_limits(table.to_numpy())
        im = ax.imshow(table.to_numpy(), aspect="auto", origin="lower",
                       extent=[x.min(), x.max(), y.min(), y.max()],
                       cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_ylabel("time")
        fig.colorbar(im, ax=ax, label=label)
    axes[1].set_xlabel("s_norm")
    savefig(fig, out_dir / "curvature_phase_heatmaps.png")


# ---------------------------------------------------------------------------
# Section moment decomposition
# ---------------------------------------------------------------------------

def plot_section_moments(section: pd.DataFrame | None, out_dir: Path) -> None:
    if section is None or section.empty:
        return
    table = section.pivot_table(index="time", columns="s_norm",
                                 values="M_sum", aggfunc="mean").sort_index()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = table.columns.to_numpy(dtype=float)
    y = table.index.to_numpy(dtype=float)
    vmin, vmax = symmetric_limits(table.to_numpy())
    im = axes[0].imshow(table.to_numpy(), aspect="auto", origin="lower",
                        extent=[x.min(), x.max(), y.min(), y.max()],
                        cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[0].set_title("Section moment M_sum(s,t)")
    axes[0].set_ylabel("time")
    fig.colorbar(im, ax=axes[0], label="M_sum")

    times = np.sort(section["time"].dropna().unique())
    sample_times = nearest_times(times, [times[0], 0.5 * times[-1], times[-1]])
    for t in sample_times:
        frame = section[np.isclose(section["time"], t)].sort_values("s_norm")
        # M_passive here is the combined passive+eb_passive composite (last column of that name)
        for col, style in [
            ("M_passive", "-"),
            ("M_eb_passive", "--"),
            ("M_damping", ":"),
            ("M_active", "-."),
            ("M_resist", (0, (3, 1, 1, 1))),
            ("M_sum", (0, (5, 1))),
        ]:
            if col in frame.columns:
                axes[1].plot(frame["s_norm"], frame[col], style, label=f"{col} t={t:.3g}")
    axes[1].set_title("Section moment snapshots")
    axes[1].set_xlabel("s_norm")
    axes[1].set_ylabel("moment")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)
    savefig(fig, out_dir / "section_moments.png")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def read_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")[:max_chars]


def write_summary(
    case_dir: Path,
    out_dir: Path,
    midline: pd.DataFrame | None,
    curvature: pd.DataFrame | None,
    force: pd.DataFrame | None,
    direction: pd.DataFrame | None,
    geometry: pd.DataFrame | None,
    section: pd.DataFrame | None,
) -> None:
    lines: list[str] = []
    metadata = read_text(case_dir / "run_metadata.txt")
    exit_status = read_text(case_dir / "exit_status.txt").strip()
    if metadata:
        lines += ["Run metadata", metadata.strip(), ""]
    if exit_status:
        lines += [f"Exit status: {exit_status}", ""]

    # Motion summary from direction_debug (preferred) or force_decomp fallback
    src = direction if (direction is not None and not direction.empty) else \
          (dedupe_time_rows(force) if force is not None and not force.empty else None)
    if src is not None and not src.empty:
        t0, t1 = finite_min(src["time"]), finite_max(src["time"])
        x0, x1 = float(src["x_cm"].iloc[0]), float(src["x_cm"].iloc[-1])
        y0, y1 = float(src["y_cm"].iloc[0]), float(src["y_cm"].iloc[-1])
        lines += [
            "Motion summary",
            f"time range: {fmt(t0)} to {fmt(t1)}",
            f"x_cm drift: {fmt(x1 - x0)} ({fmt(x0)} -> {fmt(x1)})",
            f"y_cm drift: {fmt(y1 - y0)} ({fmt(y0)} -> {fmt(y1)})",
        ]
        for col in ["v_forward", "v_fwd_cycle_avg", "a_forward", "tail_A_norm",
                    "F_CM_fwd", "cm_impulse_fwd"]:
            if col in src:
                lines.append(f"max |{col}|: {fmt(max_abs(src[col]))}")
        lines.append("")

    if force is not None and not force.empty:
        lines += ["Force/power summary"]
        for col in ["F_total_forward_on_fluid", "P_IB_on_fluid",
                    "P_passive", "P_eb_passive", "P_damping", "P_active", "P_sum",
                    "W_IB_on_fluid", "W_passive", "W_eb_passive",
                    "W_damping", "W_active", "W_sum"]:
            if col in force:
                lines.append(f"max |{col}|: {fmt(max_abs(force[col]))}")
        for col in ["dominant_L1_component", "dominant_Pabs_component"]:
            if col in force:
                lines.append(f"final {col}: {force[col].iloc[-1]}")
        lines.append("")

    if geometry is not None and not geometry.empty:
        lines += ["Geometry conservation summary"]
        for col in ["area_rel_error", "area_abs_rel_error",
                    "J_rms_error", "J_max_abs_error"]:
            if col in geometry:
                lines.append(f"max |{col}|: {fmt(max_abs(geometry[col]))}")
        if "J_min" in geometry:
            lines.append(f"J_min global min: {fmt(finite_min(geometry['J_min']))}")
        if "J_max" in geometry:
            lines.append(f"J_max global max: {fmt(finite_max(geometry['J_max']))}")
        lines.append("")

    if midline is not None and not midline.empty:
        lines += [
            "Midline summary",
            f"rows: {len(midline)}",
            f"time samples: {midline['time'].nunique()}",
        ]
        for col in ["y_lab", "y_prop", "curvature", "theta_body"]:
            if col in midline:
                lines.append(f"max |{col}|: {fmt(max_abs(midline[col]))}")
        lines.append("")

    if curvature is not None and not curvature.empty:
        lines += [
            "Curvature/phase summary",
            f"rows: {len(curvature)}",
        ]
        for col in ["kappa_body", "activation_drive", "active_moment",
                    "active_power", "effective_active_power",
                    "traveling_wave_index", "signed_traveling_wave_index",
                    "drive_following_index"]:
            if col in curvature:
                lines.append(f"max |{col}|: {fmt(max_abs(curvature[col]))}")
        lines.append("")

    if section is not None and not section.empty:
        lines += [
            "Section moment summary",
            f"rows: {len(section)}",
        ]
        for col in ["N_passive", "N_eb_passive", "N_damping", "N_active", "N_sum",
                    "M_passive", "M_eb_passive", "M_damping", "M_active", "M_sum",
                    "M_resist"]:
            if col in section:
                lines.append(f"max |{col}|: {fmt(max_abs(section[col]))}")
        lines.append("")

    out_path = out_dir / "analysis_summary.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[summary] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case_dir",
        nargs="?",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Run directory containing diagnostic CSV files. Defaults to this script's directory.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="Output directory for plots. Defaults to CASE_DIR/analysis_plots.",
    )
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    out_dir = (args.out_dir.resolve() if args.out_dir else case_dir / "analysis_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {key: read_csv(case_dir, key) for key in CSV_FILES}

    plot_run_overview(data["direction"], data["force"], data["geometry"], data["curvature"], out_dir)
    plot_direction_debug(data["direction"], out_dir)
    plot_force_power(data["force"], out_dir)
    plot_midline_snapshots(data["midline"], out_dir)
    plot_midline_heatmaps(data["midline"], out_dir)
    plot_curvature_phase(data["curvature"], out_dir)
    plot_section_moments(data["section"], out_dir)
    write_summary(
        case_dir, out_dir,
        data["midline"], data["curvature"], data["force"],
        data["direction"], data["geometry"], data["section"],
    )

    print(f"\nDone. Open PNG files in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
