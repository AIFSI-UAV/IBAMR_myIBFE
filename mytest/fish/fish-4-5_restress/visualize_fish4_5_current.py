#!/usr/bin/env python3
"""Post-process one fish4-5_1(3) IBFE run directory.

This script is adapted for the current fish4-5_1(3).cpp diagnostics:

  * active internal bending moment
  * EB/KV passive bending
  * weak dev/dil mesh stabilization
  * section-local shape stabilization

It reads the run CSV files when present, generates PNG figures, and writes an
analysis_summary.txt file with extra checks useful for diagnosing tail-mesh
explosion: J_min, dominant power components, section resultants, EBKV/active
moment recovery, and shape-stabilization contamination.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CSV file map
# ---------------------------------------------------------------------------
CSV_FILES = {
    "midline":   "midline_history.csv",
    "curvature": "curvature_phase_diag.csv",
    "force":     "force_decomposition_diag.csv",
    "direction": "direction_debug.csv",
    "geometry":  "geometry_conservation_diag.csv",
    "section":   "section_moment_decomposition.csv",
}

# Columns that should be numeric if present.  The script is intentionally
# tolerant: missing columns are skipped, so it can still read partial runs.
REQUIRED_NUMERIC_COLUMNS = {
    "midline": [
        "time", "cycle_phase", "station", "s_norm", "x_tail_to_head",
        "x_lab", "y_lab", "x_cm", "y_cm", "theta_body", "x_body", "y_body",
        "y_prop", "curvature",
    ],
    "curvature": [
        "step", "time", "cycle", "phase", "s_norm", "x_tail_to_head",
        "x_lab_norm", "y_lab_norm", "y_prop_norm", "x_body_norm", "y_body_norm",
        "A_body_norm", "h_norm", "kappa_body", "kappa_body_L",
        "activation_drive", "active_moment", "active_stress_section_moment",
        "curvature_conjugate_moment", "curvature_phase", "activation_phase",
        "phase_lag", "traveling_wave_index", "signed_traveling_wave_index",
        "drive_following_index", "theta_body", "x_cm", "y_cm",
        "active_power", "effective_active_power",
    ],
    "force": [
        "step", "time", "dt_eff", "forward_sign", "x_cm", "y_cm",
        "vcm_x", "vcm_y", "v_forward", "F_total_forward_on_fluid",
        "F_total_forward_on_fish", "P_IB_on_fluid", "P_IB_on_fish",
    ],
    "direction": [
        "step", "time", "dt_eff", "forward_sign", "x_cm", "y_cm", "x_forward",
        "vcm_x", "vcm_y", "v_forward", "v_lateral", "a_forward",
        "v_fwd_cycle_avg", "fish_area", "F_CM_fwd", "cm_impulse_fwd",
        "F_IB_on_fluid_fwd", "F_IB_on_fish_fwd", "F_IB_on_fish_lat",
        "F_IB_impulse_fwd", "F_IB_work_fwd", "tail_A_norm",
    ],
    "geometry": [
        "step", "time", "reference_area", "current_area", "current_area_abs",
        "area_rel_error", "area_abs_rel_error", "J_min", "J_max", "J_mean",
        "J_rms_error", "J_max_abs_error",
    ],
    "section": [
        "step", "time", "bin", "s_norm", "x_tail_to_head", "s_mean",
        "x_ref_mean", "h_mean", "c1_mean", "area", "ds_bin",
        "M_stabilization", "M_resist", "M_EB_model", "M_KV_model",
        "M_EBKV_model", "M_EBKV_error", "M_active_model", "M_active_error",
        "R_active_resist",
    ],
}

# Current fish4-5_1(3) force components from force_decomp_component_name().
CURRENT_COMPONENTS = ["dev", "dil", "damping", "ebkv", "shape", "active", "sum"]
LEGACY_COMPONENTS = ["passive", "eb_passive", "damping", "active", "sum"]


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
    except TypeError:  # pandas < 1.3 fallback
        df = pd.read_csv(path, low_memory=False, error_bad_lines=False, warn_bad_lines=True)

    # Convert all known numeric-looking columns; object columns such as dominant
    # component labels remain strings.
    numeric_candidates = set(REQUIRED_NUMERIC_COLUMNS.get(key, []))
    numeric_candidates.update(c for c in df.columns if c.startswith((
        "F_", "P_", "W_", "N_", "M_", "R_", "J_",
    )))
    numeric_candidates.update(c for c in df.columns if c in {
        "step", "time", "dt_eff", "bin", "s_norm", "x_tail_to_head", "area",
        "ds_bin", "x_cm", "y_cm", "v_forward", "tail_A_norm", "cycle", "phase",
    })
    for col in sorted(numeric_candidates.intersection(df.columns)):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Only drop rows when core columns that identify a sample are invalid.  Do
    # not drop due to optional diagnostic columns: explosion runs often end with
    # partially written rows.
    core = [c for c in ("time", "s_norm") if c in df.columns and key in {"midline", "curvature", "section"}]
    if key in {"force", "direction", "geometry"}:
        core = [c for c in ("time",) if c in df.columns]
    if core:
        before = len(df)
        df = df.dropna(subset=core).copy()
        dropped = before - len(df)
        if dropped:
            print(f"[clean] dropped {dropped} incomplete core rows from {path.name}")
    return df


def numeric(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def paired_numeric(df: pd.DataFrame, xcol: str, ycol: str) -> tuple[np.ndarray, np.ndarray]:
    if df is None or df.empty or xcol not in df.columns or ycol not in df.columns:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def max_abs_series(df: pd.DataFrame | None, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return math.nan
    arr = numeric(df[col])
    return float(np.max(np.abs(arr))) if arr.size else math.nan


def finite_min_series(df: pd.DataFrame | None, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return math.nan
    arr = numeric(df[col])
    return float(np.min(arr)) if arr.size else math.nan


def finite_max_series(df: pd.DataFrame | None, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return math.nan
    arr = numeric(df[col])
    return float(np.max(arr)) if arr.size else math.nan


def fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.6e}"


def dedupe_time_rows(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty or "time" not in df.columns:
        return pd.DataFrame()
    return df.sort_values("time").groupby("time", as_index=False).first().sort_values("time")


def nearest_times(times: np.ndarray, requested: Iterable[float]) -> list[float]:
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
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lim = float(np.nanpercentile(np.abs(finite), percentile))
    if lim <= 0 or not np.isfinite(lim):
        lim = float(np.nanmax(np.abs(finite)))
    if lim <= 0 or not np.isfinite(lim):
        lim = 1.0
    return -lim, lim


def zoom_limits(values: np.ndarray, pad_fraction: float = 0.15,
                min_abs_pad: float | None = None) -> tuple[float, float] | None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    span = hi - lo
    scale = max(abs(lo), abs(hi), 1.0)
    if not np.isfinite(span) or span <= 0.0:
        pad = min_abs_pad if min_abs_pad is not None else 1.0e-6 * scale
    else:
        pad = pad_fraction * span
        if min_abs_pad is not None:
            pad = max(pad, min_abs_pad)
    if not np.isfinite(pad) or pad <= 0.0:
        pad = 1.0e-6 * scale
    return lo - pad, hi + pad


def apply_zoom_ylim(ax, values: np.ndarray, pad_fraction: float = 0.15) -> None:
    lim = zoom_limits(values, pad_fraction=pad_fraction)
    if lim is not None:
        ax.set_ylim(*lim)


def savefig(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] {out_path.name}")


def plot_existing_cols(ax, df: pd.DataFrame | None, cols: Iterable[str], *, label_prefix: str = "") -> None:
    if df is None or df.empty or "time" not in df.columns:
        return
    for col in cols:
        if col in df.columns and numeric(df[col]).size:
            label = f"{label_prefix}{col}" if label_prefix else col
            ax.plot(df["time"], df[col], label=label)


def available_components(force: pd.DataFrame | None) -> list[str]:
    if force is None or force.empty:
        return CURRENT_COMPONENTS
    cols = set(force.columns)
    comps = [c for c in CURRENT_COMPONENTS if f"P_{c}" in cols or f"P_abs_{c}" in cols]
    if not comps:
        comps = [c for c in LEGACY_COMPONENTS if f"P_{c}" in cols or f"P_abs_{c}" in cols]
    return comps or CURRENT_COMPONENTS


def first_nonempty_com_source(direction: pd.DataFrame | None,
                              force: pd.DataFrame | None,
                              curvature: pd.DataFrame | None = None) -> pd.DataFrame:
    candidates = []
    if direction is not None and not direction.empty:
        candidates.append(direction)
    if force is not None and not force.empty:
        candidates.append(dedupe_time_rows(force))
    if curvature is not None and not curvature.empty:
        candidates.append(dedupe_time_rows(curvature))
    for df in candidates:
        if {"time", "x_cm", "y_cm"}.issubset(df.columns):
            return df.sort_values("time").copy()
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Overview and explosion triage
# ---------------------------------------------------------------------------

def plot_run_overview(direction: pd.DataFrame | None,
                      force: pd.DataFrame | None,
                      geometry: pd.DataFrame | None,
                      curvature: pd.DataFrame | None,
                      out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()

    src = first_nonempty_com_source(direction, force, curvature)
    if not src.empty:
        plot_existing_cols(axes[0], src, ["x_cm", "y_cm", "v_forward", "v_fwd_cycle_avg"])
        axes[0].set_title("COM and forward speed")
        axes[0].set_xlabel("time")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
    else:
        axes[0].axis("off")

    if geometry is not None and not geometry.empty:
        plot_existing_cols(axes[1], geometry, ["area_abs_rel_error", "J_rms_error", "J_max_abs_error"])
        axes[1].set_yscale("log")
        axes[1].set_title("Geometry error")
        axes[1].set_xlabel("time")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)
    else:
        axes[1].axis("off")

    if force is not None and not force.empty:
        comps = available_components(force)
        cols = [f"P_abs_{c}" for c in comps if f"P_abs_{c}" in force.columns]
        plot_existing_cols(axes[2], force, cols)
        if cols:
            axes[2].set_yscale("log")
        axes[2].set_title("Power magnitude by component")
        axes[2].set_xlabel("time")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=7, ncol=2)
    else:
        axes[2].axis("off")

    scalar = dedupe_time_rows(curvature)
    if not scalar.empty:
        plot_existing_cols(axes[3], scalar, [
            "traveling_wave_index", "signed_traveling_wave_index", "drive_following_index",
            "active_power", "effective_active_power",
        ])
        axes[3].set_title("Wave and active-power indices")
        axes[3].set_xlabel("time")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(fontsize=7)
    else:
        axes[3].axis("off")

    savefig(fig, out_dir / "run_overview_fish4_5.png")


def plot_explosion_triage(force: pd.DataFrame | None,
                          geometry: pd.DataFrame | None,
                          section: pd.DataFrame | None,
                          out_dir: Path) -> None:
    """Plots most useful for immediate tail-blow-up diagnosis."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()

    if geometry is not None and not geometry.empty:
        plot_existing_cols(axes[0], geometry, ["J_min", "J_max", "J_mean"])
        axes[0].axhline(0.0, linestyle="--", linewidth=0.8)
        axes[0].set_title("J positivity check")
        axes[0].set_xlabel("time")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
    else:
        axes[0].axis("off")

    if force is not None and not force.empty:
        cols = [c for c in [
            "P_abs_active", "P_abs_ebkv", "P_abs_shape", "P_abs_dev", "P_abs_dil", "P_abs_damping",
        ] if c in force.columns]
        plot_existing_cols(axes[1], force, cols)
        if cols:
            axes[1].set_yscale("log")
        axes[1].set_title("Explosion-source power magnitudes")
        axes[1].set_xlabel("time")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=7, ncol=2)
    else:
        axes[1].axis("off")

    if force is not None and not force.empty:
        cols = [c for c in [
            "F_weak_shape_forward_on_fish", "F_weak_ebkv_forward_on_fish",
            "F_weak_active_forward_on_fish", "F_weak_dev_forward_on_fish",
            "F_weak_dil_forward_on_fish",
        ] if c in force.columns]
        plot_existing_cols(axes[2], force, cols)
        axes[2].set_title("Forward force on fish by component")
        axes[2].set_xlabel("time")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=7)
    else:
        axes[2].axis("off")

    if section is not None and not section.empty:
        scalar = section.groupby("time", as_index=False).agg({
            c: "max" for c in ["M_active_error", "M_EBKV_error", "M_shape", "M_stabilization"] if c in section.columns
        })
        plot_existing_cols(axes[3], scalar, [
            "M_active_error", "M_EBKV_error", "M_shape", "M_stabilization",
        ])
        axes[3].set_title("Section moment errors / stabilization")
        axes[3].set_xlabel("time")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(fontsize=7)
    else:
        axes[3].axis("off")

    savefig(fig, out_dir / "explosion_triage.png")


# ---------------------------------------------------------------------------
# COM / direction
# ---------------------------------------------------------------------------

def plot_com_motion_split(direction: pd.DataFrame | None,
                          force: pd.DataFrame | None,
                          curvature: pd.DataFrame | None,
                          out_dir: Path) -> None:
    src = first_nonempty_com_source(direction, force, curvature)
    if src.empty:
        return
    for col, fname in [("x_cm", "com_x_cm_vs_time_zoom.png"), ("y_cm", "com_y_cm_vs_time_zoom.png")]:
        t, y = paired_numeric(src, "time", col)
        if t.size == 0:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.8))
        ax.plot(t, y, label=col)
        ax.axhline(y[0], linestyle="--", linewidth=0.8, label=f"initial {col}")
        drift = y[-1] - y[0]
        ax.set_title(f"{col} vs time; drift={drift:.6e}")
        ax.set_xlabel("time")
        ax.set_ylabel(col)
        apply_zoom_ylim(ax, y, pad_fraction=0.20)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        savefig(fig, out_dir / fname)

    tx, x = paired_numeric(src, "time", "x_cm")
    ty, y = paired_numeric(src, "time", "y_cm")
    if tx.size and ty.size and tx.size == ty.size and np.allclose(tx, ty):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.8))
        x_rel = x - x[0]
        y_rel = y - y[0]
        ax.plot(tx, x_rel, label="x_cm - x_cm(0)")
        ax.plot(tx, y_rel, label="y_cm - y_cm(0)")
        ax.set_title("COM relative drift vs time")
        ax.set_xlabel("time")
        ax.set_ylabel("relative displacement")
        apply_zoom_ylim(ax, np.concatenate([x_rel, y_rel]), pad_fraction=0.20)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        savefig(fig, out_dir / "com_relative_drift_vs_time.png")


def plot_direction_debug(direction: pd.DataFrame | None, out_dir: Path) -> None:
    if direction is None or direction.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plot_existing_cols(axes[0], direction, ["v_forward", "v_lateral", "v_fwd_cycle_avg"])
    axes[0].set_title("Velocity components")
    axes[0].set_ylabel("speed")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    plot_existing_cols(axes[1], direction, ["a_forward", "F_CM_fwd", "cm_impulse_fwd"])
    axes[1].set_title("CM acceleration / impulse")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plot_existing_cols(axes[2], direction, [
        "F_IB_on_fluid_fwd", "F_IB_on_fish_fwd", "F_IB_on_fish_lat", "tail_A_norm",
    ])
    axes[2].set_title("IB force and tail amplitude")
    axes[2].set_xlabel("time")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)
    savefig(fig, out_dir / "direction_debug.png")


# ---------------------------------------------------------------------------
# Force / power decomposition
# ---------------------------------------------------------------------------

def plot_force_power(force: pd.DataFrame | None, out_dir: Path) -> None:
    if force is None or force.empty:
        return
    comps = available_components(force)
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    signed_cols = ["P_IB_on_fluid", "P_IB_on_fish"] + [f"P_{c}" for c in comps if f"P_{c}" in force.columns]
    plot_existing_cols(axes[0], force, signed_cols)
    axes[0].set_title("Signed power: internal convention and IB power")
    axes[0].set_ylabel("power")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)

    abs_cols = [f"P_abs_{c}" for c in comps if f"P_abs_{c}" in force.columns]
    plot_existing_cols(axes[1], force, abs_cols)
    if abs_cols:
        axes[1].set_yscale("log")
    axes[1].set_title("Absolute power by component")
    axes[1].set_ylabel("abs power")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)

    work_cols = [
        "W_IB_on_fluid", "W_IB_on_fish", "W_dev", "W_dil", "W_damping", "W_ebkv", "W_shape", "W_active", "W_sum",
        "W_passive", "W_eb_passive",  # legacy fallback
    ]
    plot_existing_cols(axes[2], force, [c for c in work_cols if c in force.columns])
    axes[2].set_title("Accumulated work")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("work")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=7, ncol=2)
    savefig(fig, out_dir / "force_power_components.png")


def plot_force_forward_components(force: pd.DataFrame | None, out_dir: Path) -> None:
    if force is None or force.empty:
        return
    comps = available_components(force)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    fluid_cols = ["F_total_forward_on_fluid"] + [
        f"F_weak_{c}_forward_on_fluid" for c in comps if f"F_weak_{c}_forward_on_fluid" in force.columns
    ]
    fish_cols = ["F_total_forward_on_fish"] + [
        f"F_weak_{c}_forward_on_fish" for c in comps if f"F_weak_{c}_forward_on_fish" in force.columns
    ]
    plot_existing_cols(axes[0], force, fluid_cols)
    axes[0].set_title("Forward force on fluid")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)

    plot_existing_cols(axes[1], force, fish_cols)
    axes[1].set_title("Forward force on fish")
    axes[1].set_xlabel("time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    savefig(fig, out_dir / "force_forward_components.png")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def plot_geometry(geometry: pd.DataFrame | None, out_dir: Path) -> None:
    if geometry is None or geometry.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    plot_existing_cols(axes[0], geometry, ["J_min", "J_max", "J_mean"])
    axes[0].axhline(0.0, linestyle="--", linewidth=0.8)
    axes[0].set_title("Jacobian metrics")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    plot_existing_cols(axes[1], geometry, ["area_rel_error", "area_abs_rel_error", "J_rms_error", "J_max_abs_error"])
    axes[1].set_yscale("log")
    axes[1].set_title("Geometry conservation errors")
    axes[1].set_xlabel("time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    savefig(fig, out_dir / "geometry_conservation.png")


# ---------------------------------------------------------------------------
# Midline
# ---------------------------------------------------------------------------

def plot_midline_snapshots(midline: pd.DataFrame | None, out_dir: Path) -> None:
    if midline is None or midline.empty:
        return
    times = np.sort(midline["time"].dropna().unique())
    if times.size == 0:
        return
    sample_times = nearest_times(times, [times[0], 0.25 * times[-1], 0.5 * times[-1], 0.75 * times[-1], times[-1]])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    y_lab_samples: list[np.ndarray] = []
    x_lab_samples: list[np.ndarray] = []
    y_body_samples: list[np.ndarray] = []
    curv_samples: list[np.ndarray] = []
    body_y_col = "y_body" if "y_body" in midline.columns else ("y_prop" if "y_prop" in midline.columns else "y_lab")

    for t in sample_times:
        frame = midline[np.isclose(midline["time"], t)].sort_values("s_norm")
        if frame.empty:
            continue
        if {"x_lab", "y_lab"}.issubset(frame.columns):
            x = pd.to_numeric(frame["x_lab"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(frame["y_lab"], errors="coerce").to_numpy(dtype=float)
            axes[0].plot(x, y, label=f"t={t:.3g}")
            x_lab_samples.append(x[np.isfinite(x)])
            y_lab_samples.append(y[np.isfinite(y)])
        if {"s_norm", body_y_col}.issubset(frame.columns):
            axes[1].plot(frame["s_norm"], frame[body_y_col], label=f"t={t:.3g}")
            yb = pd.to_numeric(frame[body_y_col], errors="coerce").to_numpy(dtype=float)
            y_body_samples.append(yb[np.isfinite(yb)])
        if {"s_norm", "curvature"}.issubset(frame.columns):
            axes[2].plot(frame["s_norm"], frame["curvature"], label=f"t={t:.3g}")
            cv = pd.to_numeric(frame["curvature"], errors="coerce").to_numpy(dtype=float)
            curv_samples.append(cv[np.isfinite(cv)])

    axes[0].set_title("Midline lab frame, zoomed y")
    axes[0].set_xlabel("x_lab")
    axes[0].set_ylabel("y_lab")
    if x_lab_samples:
        xlim = zoom_limits(np.concatenate(x_lab_samples), pad_fraction=0.03)
        if xlim is not None:
            axes[0].set_xlim(*xlim)
    if y_lab_samples:
        apply_zoom_ylim(axes[0], np.concatenate(y_lab_samples), pad_fraction=0.25)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title(f"Body-frame lateral motion {body_y_col}(s,t)")
    axes[1].set_xlabel("s_norm")
    axes[1].set_ylabel(body_y_col)
    if y_body_samples:
        apply_zoom_ylim(axes[1], np.concatenate(y_body_samples), pad_fraction=0.25)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].set_title("Curvature snapshots")
    axes[2].set_xlabel("s_norm")
    axes[2].set_ylabel("curvature")
    if curv_samples:
        lo, hi = symmetric_limits(np.concatenate(curv_samples), percentile=99.5)
        axes[2].set_ylim(lo, hi)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)
    savefig(fig, out_dir / "midline_snapshots_zoomed.png")


def plot_midline_heatmaps(midline: pd.DataFrame | None, out_dir: Path) -> None:
    if midline is None or midline.empty or "time" not in midline.columns or "s_norm" not in midline.columns:
        return
    value_cols = [c for c in ["curvature", "y_prop", "y_body", "y_lab"] if c in midline.columns]
    if not value_cols:
        return
    n = len(value_cols)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.8 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, value_cols):
        table = midline.pivot_table(index="time", columns="s_norm", values=col, aggfunc="mean").sort_index()
        if table.empty:
            ax.axis("off")
            continue
        x = table.columns.to_numpy(dtype=float)
        y = table.index.to_numpy(dtype=float)
        vmin, vmax = symmetric_limits(table.to_numpy())
        im = ax.imshow(table.to_numpy(), aspect="auto", origin="lower",
                       extent=[x.min(), x.max(), y.min(), y.max()], vmin=vmin, vmax=vmax)
        ax.set_title(f"{col}(s,t)")
        ax.set_ylabel("time")
        fig.colorbar(im, ax=ax, label=col)
    axes[-1].set_xlabel("s_norm")
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

    plot_existing_cols(axes[0], scalar, ["x_cm", "y_cm", "y_cm_relative"])
    axes[0].set_title("COM from curvature diagnostics")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    plot_existing_cols(axes[1], scalar, ["curvature_phase", "activation_phase", "phase_lag", "curvature_phase_unwrapped"])
    axes[1].set_title("Phase diagnostics")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plot_existing_cols(axes[2], scalar, ["curvature_phase_slope", "active_phase_slope_abs", "active_phase_slope_expected"])
    axes[2].set_title("Phase slopes")
    axes[2].set_xlabel("time")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    plot_existing_cols(axes[3], scalar, [
        "raw_active_power", "active_power", "effective_active_power",
        "active_signed_work", "effective_active_signed_work",
    ])
    axes[3].set_title("Active power / work")
    axes[3].set_xlabel("time")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=8)
    savefig(fig, out_dir / "curvature_phase_diagnostics.png")

    # Heatmaps for most relevant fields.
    heat_cols = [c for c in ["kappa_body", "activation_drive", "active_moment", "A_body_norm"] if c in curvature.columns]
    if not heat_cols or "s_norm" not in curvature.columns:
        return
    fig, axes = plt.subplots(len(heat_cols), 1, figsize=(12, 3.7 * len(heat_cols)), sharex=True)
    if len(heat_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, heat_cols):
        table = curvature.pivot_table(index="time", columns="s_norm", values=col, aggfunc="mean").sort_index()
        if table.empty:
            ax.axis("off")
            continue
        x = table.columns.to_numpy(dtype=float)
        y = table.index.to_numpy(dtype=float)
        vmin, vmax = symmetric_limits(table.to_numpy())
        im = ax.imshow(table.to_numpy(), aspect="auto", origin="lower",
                       extent=[x.min(), x.max(), y.min(), y.max()], vmin=vmin, vmax=vmax)
        ax.set_title(f"{col}(s,t)")
        ax.set_ylabel("time")
        fig.colorbar(im, ax=ax, label=col)
    axes[-1].set_xlabel("s_norm")
    savefig(fig, out_dir / "curvature_phase_heatmaps.png")


# ---------------------------------------------------------------------------
# Section moment/resultant diagnostics
# ---------------------------------------------------------------------------

def plot_section_moments(section: pd.DataFrame | None, out_dir: Path) -> None:
    if section is None or section.empty or "s_norm" not in section.columns:
        return

    # Heatmaps for key moment channels.
    heat_cols = [c for c in [
        "M_sum", "M_active", "M_ebkv", "M_shape", "M_stabilization", "M_EBKV_error", "M_active_error",
    ] if c in section.columns]
    if heat_cols:
        fig, axes = plt.subplots(len(heat_cols), 1, figsize=(12, 3.2 * len(heat_cols)), sharex=True)
        if len(heat_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, heat_cols):
            table = section.pivot_table(index="time", columns="s_norm", values=col, aggfunc="mean").sort_index()
            if table.empty:
                ax.axis("off")
                continue
            x = table.columns.to_numpy(dtype=float)
            y = table.index.to_numpy(dtype=float)
            vmin, vmax = symmetric_limits(table.to_numpy(), percentile=99.5)
            im = ax.imshow(table.to_numpy(), aspect="auto", origin="lower",
                           extent=[x.min(), x.max(), y.min(), y.max()], vmin=vmin, vmax=vmax)
            ax.set_title(f"Section {col}(s,t)")
            ax.set_ylabel("time")
            fig.colorbar(im, ax=ax, label=col)
        axes[-1].set_xlabel("s_norm")
        savefig(fig, out_dir / "section_moment_heatmaps.png")

    times = np.sort(section["time"].dropna().unique())
    if times.size == 0:
        return
    sample_times = nearest_times(times, [times[0], 0.5 * times[-1], times[-1]])

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    moment_cols = [c for c in [
        "M_active_model", "M_active", "M_ebkv", "M_EB_model", "M_KV_model",
        "M_shape", "M_stabilization", "M_resist", "M_sum",
    ] if c in section.columns]
    for t in sample_times:
        frame = section[np.isclose(section["time"], t)].sort_values("s_norm")
        for col in moment_cols:
            axes[0].plot(frame["s_norm"], frame[col], label=f"{col} t={t:.3g}")
    axes[0].set_title("Section moment snapshots")
    axes[0].set_ylabel("moment")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=6, ncol=3)

    resultant_cols = [c for c in ["N_active", "N_ebkv", "N_shape", "N_dev", "N_dil", "N_sum"] if c in section.columns]
    for t in sample_times:
        frame = section[np.isclose(section["time"], t)].sort_values("s_norm")
        for col in resultant_cols:
            axes[1].plot(frame["s_norm"], frame[col], label=f"{col} t={t:.3g}")
    axes[1].set_title("Section axial resultants")
    axes[1].set_xlabel("s_norm")
    axes[1].set_ylabel("resultant")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=6, ncol=3)
    savefig(fig, out_dir / "section_moment_snapshots.png")


def plot_section_error_timeseries(section: pd.DataFrame | None, out_dir: Path) -> None:
    if section is None or section.empty:
        return
    agg_map = {}
    for col in [
        "M_active_error", "M_EBKV_error", "N_active", "N_ebkv", "N_shape",
        "M_shape", "M_stabilization", "R_active_resist",
    ]:
        if col in section.columns:
            agg_map[col] = lambda s: float(np.nanmax(np.abs(pd.to_numeric(s, errors="coerce"))))
    if not agg_map:
        return
    # Avoid late-binding lambda issue by manual aggregation.
    rows = []
    for t, group in section.groupby("time"):
        row = {"time": t}
        for col in agg_map:
            arr = numeric(group[col])
            row[f"max_abs_{col}"] = float(np.max(np.abs(arr))) if arr.size else math.nan
        rows.append(row)
    ts = pd.DataFrame(rows).sort_values("time")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for col in ts.columns:
        if col == "time":
            continue
        if numeric(ts[col]).size:
            ax.plot(ts["time"], ts[col], label=col)
    ax.set_yscale("log")
    ax.set_title("Max absolute section errors/resultants vs time")
    ax.set_xlabel("time")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    savefig(fig, out_dir / "section_error_timeseries.png")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def read_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")[:max_chars]


def component_power_ratios(force: pd.DataFrame | None) -> list[str]:
    if force is None or force.empty:
        return []
    lines: list[str] = []
    comps = [c for c in available_components(force) if c != "sum"]
    active_max = max_abs_series(force, "P_abs_active")
    if not np.isfinite(active_max) or active_max <= 0:
        active_max = max_abs_series(force, "P_active")
    for c in comps:
        pabs = max_abs_series(force, f"P_abs_{c}")
        if np.isfinite(pabs):
            ratio = pabs / active_max if np.isfinite(active_max) and active_max > 0 else math.nan
            lines.append(f"max P_abs_{c}: {fmt(pabs)}; ratio to active: {fmt(ratio)}")
    return lines


def write_summary(case_dir: Path,
                  out_dir: Path,
                  data: dict[str, pd.DataFrame | None]) -> None:
    midline = data.get("midline")
    curvature = data.get("curvature")
    force = data.get("force")
    direction = data.get("direction")
    geometry = data.get("geometry")
    section = data.get("section")

    lines: list[str] = []
    metadata = read_text(case_dir / "run_metadata.txt")
    exit_status = read_text(case_dir / "exit_status.txt").strip()
    if metadata:
        lines += ["Run metadata", metadata.strip(), ""]
    if exit_status:
        lines += [f"Exit status: {exit_status}", ""]

    src = first_nonempty_com_source(direction, force, curvature)
    if not src.empty:
        t0, t1 = finite_min_series(src, "time"), finite_max_series(src, "time")
        x = numeric(src["x_cm"])
        y = numeric(src["y_cm"])
        lines += ["Motion summary", f"time range: {fmt(t0)} to {fmt(t1)}"]
        if x.size:
            lines.append(f"x_cm drift: {fmt(x[-1] - x[0])} ({fmt(x[0])} -> {fmt(x[-1])})")
        if y.size:
            lines.append(f"y_cm drift: {fmt(y[-1] - y[0])} ({fmt(y[0])} -> {fmt(y[-1])})")
        for col in ["v_forward", "v_fwd_cycle_avg", "a_forward", "tail_A_norm", "F_CM_fwd", "cm_impulse_fwd"]:
            if col in src.columns:
                lines.append(f"max |{col}|: {fmt(max_abs_series(src, col))}")
        lines.append("")

    if geometry is not None and not geometry.empty:
        lines += ["Geometry conservation summary"]
        for col in ["area_rel_error", "area_abs_rel_error", "J_rms_error", "J_max_abs_error"]:
            if col in geometry.columns:
                lines.append(f"max |{col}|: {fmt(max_abs_series(geometry, col))}")
        if "J_min" in geometry.columns:
            lines.append(f"J_min global min: {fmt(finite_min_series(geometry, 'J_min'))}")
        if "J_max" in geometry.columns:
            lines.append(f"J_max global max: {fmt(finite_max_series(geometry, 'J_max'))}")
        lines.append("")

    if force is not None and not force.empty:
        lines += ["Force / power summary"]
        for col in ["F_total_forward_on_fluid", "F_total_forward_on_fish", "P_IB_on_fluid", "P_IB_on_fish"]:
            if col in force.columns:
                lines.append(f"max |{col}|: {fmt(max_abs_series(force, col))}")
        lines += component_power_ratios(force)
        for col in ["dominant_L1_component", "dominant_Pabs_component"]:
            if col in force.columns and len(force[col]):
                lines.append(f"final {col}: {force[col].iloc[-1]}")
        lines.append("")

    if section is not None and not section.empty:
        lines += ["Section-resultant summary"]
        for col in [
            "N_active", "N_ebkv", "N_shape", "N_dev", "N_dil", "N_sum",
            "M_active", "M_ebkv", "M_shape", "M_stabilization", "M_resist", "M_sum",
            "M_EBKV_error", "M_active_error", "R_active_resist",
        ]:
            if col in section.columns:
                lines.append(f"max |{col}|: {fmt(max_abs_series(section, col))}")
        lines.append("")

    if midline is not None and not midline.empty:
        lines += ["Midline summary", f"rows: {len(midline)}", f"time samples: {midline['time'].nunique() if 'time' in midline else 'n/a'}"]
        for col in ["y_lab", "y_prop", "y_body", "curvature", "theta_body"]:
            if col in midline.columns:
                lines.append(f"max |{col}|: {fmt(max_abs_series(midline, col))}")
        lines.append("")

    if curvature is not None and not curvature.empty:
        lines += ["Curvature / phase summary", f"rows: {len(curvature)}"]
        for col in [
            "kappa_body", "activation_drive", "active_moment", "active_power", "effective_active_power",
            "traveling_wave_index", "signed_traveling_wave_index", "drive_following_index",
        ]:
            if col in curvature.columns:
                lines.append(f"max |{col}|: {fmt(max_abs_series(curvature, col))}")
        lines.append("")

    lines += [
        "Interpretation checklist for fish4-5_1(3)",
        "1. If J_min <= 0 or J_min rapidly approaches 0, the FE mesh is inverting.",
        "2. If P_abs_shape or M_shape is comparable to active/EBKV, section-shape stabilization is influencing physics, not only mesh quality.",
        "3. If P_abs_dev or P_abs_dil dominates, dev/dil are still contaminating propulsion.",
        "4. If M_active_error or M_EBKV_error is large, section moment recovery is not clean.",
        "5. For tail blow-up, first test: shape off, EBKV off, then enable them one at a time.",
    ]

    out_path = out_dir / "analysis_summary.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[summary] {out_path.name}")


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
        help="Output directory for plots. Defaults to CASE_DIR/analysis_plots_fish4_5.",
    )
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    out_dir = (args.out_dir.resolve() if args.out_dir else case_dir / "analysis_plots_fish4_5")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {key: read_csv(case_dir, key) for key in CSV_FILES}

    plot_run_overview(data["direction"], data["force"], data["geometry"], data["curvature"], out_dir)
    plot_explosion_triage(data["force"], data["geometry"], data["section"], out_dir)
    plot_com_motion_split(data["direction"], data["force"], data["curvature"], out_dir)
    plot_direction_debug(data["direction"], out_dir)
    plot_geometry(data["geometry"], out_dir)
    plot_force_power(data["force"], out_dir)
    plot_force_forward_components(data["force"], out_dir)
    plot_midline_snapshots(data["midline"], out_dir)
    plot_midline_heatmaps(data["midline"], out_dir)
    plot_curvature_phase(data["curvature"], out_dir)
    plot_section_moments(data["section"], out_dir)
    plot_section_error_timeseries(data["section"], out_dir)
    write_summary(case_dir, out_dir, data)

    print(f"\nDone. Open PNG/TXT files in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())