#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
如果你当前目录下就有：
input2d_1
fish2d.msh
fish4-4_1.cpp

python3 plot_actuation_model_time_series_mesh_consistent.py input2d_1 --mesh fish2d.msh --cpp fish4-4_1.cpp --t-end 4.0 --s-frac 0.50 0.70 0.85 --output-dir figs_time_mesh

plot_actuation_model_time_series_mesh_consistent.py

Version C: mesh-consistent, current continuum-active model
- 从 input2d 读取当前 fish4-4 参数
- 按 fish2d.msh / fish.msh 的真实边界几何重建 backbone
- 使用与当前 fish4-4_1.cpp 一致的:
    * tail-root backbone truncation
    * FE-Galerkin Laplace/harmonic coordinate when USE_LAPLACE_REFERENCE_PARAMETERIZATION=TRUE
    * strict phi-isocontour boundary sections for eta=0 and h(s)
    * centerline projection h(s) fallback only when the input disables Laplace mode
    * ACTIVE_MOMENT_MODE:
      - TRAVELING: R(t) * env_s * beta_act * h(s)^2 * K_shape(xi) * cos(phase)
      - STATIC: R(t) * STATIC_MOMENT_M0 inside the active interval
      [HALF-BELL mode: 0.5*(1-cos(pi*xi)), posterior-rising]
      [BELL mode: 1-cos(2*pi*xi), Xu/Zhou/Yu 2024 Eq. (2)]
    * xi = active-body coordinate from ACTIVE_S_START to ACTIVE_S_END
    * phase = 2*pi*xi/LAMBDA_ACT_OVER_LACT - WAVE_TIME_SIGN*omega*t + ACTIVE_PHASE0
    * env_s(s/ref_arc_length)
    * active stress saturation Mm_clamped = Mm_max*tanh(Mm_raw/Mm_max)
      with the current C++ I2 floor concept:
      I2_ideal = I2_eff_unit*h(s)^ACTIVE_I2_H_POWER and
      I2_use = max(I2_c_FE_scaled, FE_SECTION_I2_FLOOR_RATIO*I2_ideal) at runtime.
      This preprocessor reports the analytic proxy I2_use_proxy=I2_ideal
      when no FE section CSV is available.
    * REFERENCE_BACKBONE_END_X mapped through the Laplace/harmonic coordinate
      as a reference-end diagnostic, not as a cap on ACTIVE_S_END

输出:
- 00_Ks_only.png
- 01_static_profiles.png
- 02_ramp_vs_time.png
- 03_Mm_time_series_points.png
- 04_Mm_raw_heatmap.png
- 05_C1S_profile.png
- 06_K_over_h3.png
- 07_phase_carrier.png
- 08_Mm_rms_profile.png
- 09_Mm_profiles_s.png
- overview.png
- actuation_static_profile.csv
- actuation_time_series.csv
- actuation_Mm_profiles_s.csv
- README.txt
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# fish4-4_1-aligned built-in defaults
# ----------------------------------------------------------------------
DEFAULTS: Dict[str, object] = {
    "FISH_LENGTH": 1.00,
    "X_LEADING": 1.00,
    "WAVE_FREQUENCY": 1.00,
    "WAVE_RAMP_TIME": 3.0,
    "BETA_ACT": 0.5,
    "ACTIVE_MOMENT_MODE": "TRAVELING",
    "STATIC_MOMENT_M0": 0.0,
    "INITIAL_BEND_AMPLITUDE": 0.0,
    "ACTIVE_PHASE_COORDINATE": "ACTIVE_BODY_XI",
    "LAMBDA_ACT_OVER_LACT": 1.00,
    "ACTIVE_PHASE0": 0.0,
    "ACTIVE_MOMENT_TO_STRESS_SIGN": -1.0,
    "ACTIVE_S_START": 0.00,
    "ACTIVE_S_END": 1.00,
    "ACTIVE_S_SMOOTH": 0.0,
    "ACTIVE_BAND_FRACTION": 1.0,
    "ACTIVE_I2_H_POWER": 3.0,
    "ACTIVE_T_ACT_MAX_OVER_C1": 100.0,
    "FE_SECTION_I2_FLOOR_RATIO": 0.20,
    "K_SHAPE_MODE": "BELL",
    # In cos(k*xi - WAVE_TIME_SIGN*omega*t), +1 propagates head-to-tail.
    "WAVE_TIME_SIGN": 1.0,
    "C1_S_PASSIVE": 0.5,
    "C1_S_PASSIVE_ANTERIOR": 0.5,
    "C1_S_PASSIVE_PEDUNCLE": 0.5,
    "C1_S_PASSIVE_CAUDAL": 0.5,
    "C1_S_PASSIVE_BODY_TRANSITION_S": 0.60,
    "C1_S_PASSIVE_BODY_TRANSITION_W": 0.30,
    "C1_S_PASSIVE_CAUDAL_TRANSITION_S": 0.85,
    "C1_S_PASSIVE_CAUDAL_TRANSITION_W": 0.10,
    "KAPPA_VOL_PASSIVE": 20.0,
    "USE_CONTINUUM_DAMPING": False,
    "CONTINUUM_DAMPING_FACTOR": 0.005,
    "CONTINUUM_DAMPING_STRESS_CAP_OVER_C1": 50.0,
    # Legacy aliases retained so older input files can still be plotted.
    "C1_S_BODY_TRANSITION_S": 0.60,
    "C1_S_BODY_TRANSITION_W": 0.30,
    "C1_S_CAUDAL_TRANSITION_S": 0.85,
    "C1_S_CAUDAL_TRANSITION_W": 0.10,
    "REFERENCE_PROFILE_BINS": 128,
    "REFERENCE_BACKBONE_END_X": float("nan"),
    "USE_LAPLACE_REFERENCE_PARAMETERIZATION": True,
    "ALLOW_CENTERLINE_FALLBACK": False,
    "USE_FE_ACTIVE_SECTION_DATA": False,
    "LAPLACE_HEAD_BC_WIDTH_OVER_L": 0.05,
    "LAPLACE_TAIL_BC_WIDTH_OVER_L": 0.05,
    "MESH_FILENAME": "fish2d.msh",
}

CPP_NAME_MAP = {
    "fish_length": "FISH_LENGTH",
    "x_leading": "X_LEADING",
    "wave_frequency": "WAVE_FREQUENCY",
    "wave_ramp_time": "WAVE_RAMP_TIME",
    "beta_act": "BETA_ACT",
    "static_moment_m0": "STATIC_MOMENT_M0",
    "initial_bend_amplitude": "INITIAL_BEND_AMPLITUDE",
    "active_wavelength_over_L": "LAMBDA_ACT_OVER_LACT",
    "active_phase0": "ACTIVE_PHASE0",
    "active_moment_to_stress_sign": "ACTIVE_MOMENT_TO_STRESS_SIGN",
    "active_s_start": "ACTIVE_S_START",
    "active_s_end": "ACTIVE_S_END",
    "active_s_smooth": "ACTIVE_S_SMOOTH",
    "active_band_fraction": "ACTIVE_BAND_FRACTION",
    "active_i2_h_power": "ACTIVE_I2_H_POWER",
    "active_t_act_max_over_c1": "ACTIVE_T_ACT_MAX_OVER_C1",
    "fe_section_i2_floor_ratio": "FE_SECTION_I2_FLOOR_RATIO",
    "wave_time_sign": "WAVE_TIME_SIGN",
    "c1_s_passive": "C1_S_PASSIVE",
    "c1_s_passive_anterior": "C1_S_PASSIVE_ANTERIOR",
    "c1_s_passive_peduncle": "C1_S_PASSIVE_PEDUNCLE",
    "c1_s_passive_caudal": "C1_S_PASSIVE_CAUDAL",
    "c1_s_body_transition_s": "C1_S_PASSIVE_BODY_TRANSITION_S",
    "c1_s_body_transition_w": "C1_S_PASSIVE_BODY_TRANSITION_W",
    "c1_s_caudal_transition_s": "C1_S_PASSIVE_CAUDAL_TRANSITION_S",
    "c1_s_caudal_transition_w": "C1_S_PASSIVE_CAUDAL_TRANSITION_W",
    "kappa_vol": "KAPPA_VOL_PASSIVE",
    "use_continuum_damping": "USE_CONTINUUM_DAMPING",
    "continuum_damping_factor": "CONTINUUM_DAMPING_FACTOR",
    "continuum_damping_stress_cap_over_c1": "CONTINUUM_DAMPING_STRESS_CAP_OVER_C1",
    "reference_profile_bins": "REFERENCE_PROFILE_BINS",
    "reference_backbone_end_x": "REFERENCE_BACKBONE_END_X",
    "use_laplace_reference_parameterization": "USE_LAPLACE_REFERENCE_PARAMETERIZATION",
    "allow_centerline_fallback": "ALLOW_CENTERLINE_FALLBACK",
    "use_fe_active_section_data": "USE_FE_ACTIVE_SECTION_DATA",
    "laplace_head_bc_width_over_L": "LAPLACE_HEAD_BC_WIDTH_OVER_L",
    "laplace_tail_bc_width_over_L": "LAPLACE_TAIL_BC_WIDTH_OVER_L",
}

INPUT_NAME_MAP = {
    # Current names map to themselves through DEFAULTS. These entries support
    # older EB/FACTORIZED inputs without changing the current output semantics.
    "ACTIVE_MOMENT_MODEL": "ACTIVE_MOMENT_MODE",
    "C1_S_PHYSICS": "C1_S_PASSIVE",
    "C1_S_PHYSICS_ANTERIOR": "C1_S_PASSIVE_ANTERIOR",
    "C1_S_PHYSICS_PEDUNCLE": "C1_S_PASSIVE_PEDUNCLE",
    "C1_S_PHYSICS_CAUDAL": "C1_S_PASSIVE_CAUDAL",
    "C1_S_PHYSICS_BODY_TRANSITION_S": "C1_S_PASSIVE_BODY_TRANSITION_S",
    "C1_S_PHYSICS_BODY_TRANSITION_W": "C1_S_PASSIVE_BODY_TRANSITION_W",
    "C1_S_PHYSICS_CAUDAL_TRANSITION_S": "C1_S_PASSIVE_CAUDAL_TRANSITION_S",
    "C1_S_PHYSICS_CAUDAL_TRANSITION_W": "C1_S_PASSIVE_CAUDAL_TRANSITION_W",
    "C1_S": "C1_S_PASSIVE",
    "C1_S_ANTERIOR": "C1_S_PASSIVE_ANTERIOR",
    "C1_S_PEDUNCLE": "C1_S_PASSIVE_PEDUNCLE",
    "C1_S_CAUDAL": "C1_S_PASSIVE_CAUDAL",
    "C1_S_BODY_TRANSITION_S": "C1_S_PASSIVE_BODY_TRANSITION_S",
    "C1_S_BODY_TRANSITION_W": "C1_S_PASSIVE_BODY_TRANSITION_W",
    "C1_S_CAUDAL_TRANSITION_S": "C1_S_PASSIVE_CAUDAL_TRANSITION_S",
    "C1_S_CAUDAL_TRANSITION_W": "C1_S_PASSIVE_CAUDAL_TRANSITION_W",
    "KAPPA_VOL": "KAPPA_VOL_PASSIVE",
}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def smoothstep(x):
    x = clamp01(x)
    return x * x * (3.0 - 2.0 * x)


def smoothstep_cosine(s, s0: float, w: float):
    s_arr = np.asarray(s, dtype=float)
    width = max(float(w), 1.0e-12)
    lo = float(s0) - 0.5 * width
    hi = float(s0) + 0.5 * width
    xi = np.clip((s_arr - lo) / width, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * xi))


def active_band_weight_unit(q_abs, active_band_fraction: float):
    q = clamp01(np.asarray(q_abs, dtype=float))
    f = float(clamp01(active_band_fraction))
    if f <= 1.0e-12:
        return np.zeros_like(q, dtype=float)
    inner = max(0.0, 1.0 - f)
    r = clamp01((q - inner) / f)
    return np.where(q <= inner, 0.0, 0.5 * (1.0 - np.cos(np.pi * r)))


def active_band_second_moment_unit(active_band_fraction: float) -> float:
    n = 256
    dq = 2.0 / float(n)
    q = -1.0 + (np.arange(n, dtype=float) + 0.5) * dq
    w = active_band_weight_unit(np.abs(q), active_band_fraction)
    return float(np.sum(w * q * q * dq))


def point_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - a))


def try_number(text: str):
    text = text.strip()
    try:
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        return float(text)
    except Exception:
        return text


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().strip('"').upper() in ("TRUE", "1", "YES", "ON")


def normalize_k_shape_mode(value) -> str:
    raw = str(value).strip().strip('"').upper()
    mode = "".join(ch for ch in raw if not ch.isspace()).replace("_", "-")
    if mode not in ("HALF-BELL", "BELL"):
        raise ValueError(f'K_SHAPE_MODE must be "HALF-BELL" or "BELL", got {value!r}')
    return mode


def normalize_active_moment_mode(value) -> str:
    raw = str(value).strip().strip('"').upper()
    mode = "".join(ch for ch in raw if not ch.isspace()).replace("_", "-")
    if mode in ("TRAVELING", "XU2024-SIMPLE", "SIMPLE", "FACTORIZED", "LEGACY"):
        return "TRAVELING"
    if mode == "STATIC":
        return "STATIC"
    raise ValueError(
        f'ACTIVE_MOMENT_MODE must be "TRAVELING" or "STATIC", got {value!r}'
    )


def uses_static_active_moment(params: Dict[str, object]) -> bool:
    return normalize_active_moment_mode(params.get("ACTIVE_MOMENT_MODE", "TRAVELING")) == "STATIC"


# ----------------------------------------------------------------------
# input2d / cpp parsing
# ----------------------------------------------------------------------
def strip_comments(line: str) -> str:
    if "//" in line:
        line = line.split("//", 1)[0]
    if "#" in line:
        line = line.split("#", 1)[0]
    return line.strip()


def parse_input2d(path: Path) -> Dict[str, object]:
    vals: Dict[str, object] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = strip_comments(raw)
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().rstrip(";").strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        vals[key] = try_number(val)
    return vals


def parse_cpp_defaults(path: Path) -> Dict[str, object]:
    vals: Dict[str, object] = {}
    txt = path.read_text(encoding="utf-8", errors="ignore")

    pat = re.compile(r"static\s+(double|int)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);")
    for m in pat.finditer(txt):
        typ, var, rhs = m.groups()
        if var not in CPP_NAME_MAP:
            continue

        rhs = rhs.strip()
        if "numeric_limits" in rhs:
            val = float("nan")
        else:
            rhs_eval = rhs.replace("M_PI", str(math.pi))
            try:
                val = eval(rhs_eval, {"__builtins__": {}}, {})
            except Exception:
                continue

        vals[CPP_NAME_MAP[var]] = int(val) if typ == "int" else float(val)

    str_pat = re.compile(r"static\s+std::string\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\"([^\"]*)\"\s*;")
    for m in str_pat.finditer(txt):
        var, val = m.groups()
        if var in CPP_NAME_MAP:
            vals[CPP_NAME_MAP[var]] = val

    bool_pat = re.compile(r"static\s+bool\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(true|false)\s*;",
                          re.IGNORECASE)
    for m in bool_pat.finditer(txt):
        var, val = m.groups()
        if var in CPP_NAME_MAP:
            vals[CPP_NAME_MAP[var]] = val.lower() == "true"

    moment_mode_pat = re.compile(
        r"static\s+ActiveMomentMode\s+active_moment_mode\s*=\s*ActiveMomentMode::([A-Za-z_][A-Za-z0-9_]*)\s*;"
    )
    m = moment_mode_pat.search(txt)
    if m:
        vals["ACTIVE_MOMENT_MODE"] = normalize_active_moment_mode(m.group(1))

    k_shape_pat = re.compile(
        r"static\s+ActiveKShapeMode\s+active_k_shape_mode\s*=\s*ActiveKShapeMode::([A-Za-z_][A-Za-z0-9_]*)\s*;"
    )
    m = k_shape_pat.search(txt)
    if m:
        vals["K_SHAPE_MODE"] = normalize_k_shape_mode(m.group(1))

    return vals


def build_param_dict(input_vals: Dict[str, object], cpp_vals: Dict[str, object]) -> Dict[str, object]:
    params = dict(DEFAULTS)
    params.update(cpp_vals)
    for key, val in input_vals.items():
        mapped_key = INPUT_NAME_MAP.get(key, key)
        if mapped_key in params or mapped_key == "MESH_FILENAME":
            params[mapped_key] = val

    params["ACTIVE_S_START"] = float(clamp01(float(params["ACTIVE_S_START"])))
    params["ACTIVE_S_END"] = float(clamp01(float(params["ACTIVE_S_END"])))
    params["ACTIVE_S_SMOOTH"] = max(0.0, float(params["ACTIVE_S_SMOOTH"]))
    params["ACTIVE_BAND_FRACTION"] = max(0.0, float(params["ACTIVE_BAND_FRACTION"]))
    params["ACTIVE_I2_H_POWER"] = max(0.0, float(params.get("ACTIVE_I2_H_POWER", 3.0)))
    params["FE_SECTION_I2_FLOOR_RATIO"] = max(
        0.0, float(params.get("FE_SECTION_I2_FLOOR_RATIO", 0.20))
    )
    params["ACTIVE_MOMENT_TO_STRESS_SIGN"] = (
        1.0 if float(params.get("ACTIVE_MOMENT_TO_STRESS_SIGN", 1.0)) >= 0.0 else -1.0
    )
    params["ACTIVE_T_ACT_MAX_OVER_C1"] = max(
        float(params.get("ACTIVE_T_ACT_MAX_OVER_C1", 100.0)), 1.0e-12
    )
    params["ACTIVE_MOMENT_MODE"] = normalize_active_moment_mode(
        params.get("ACTIVE_MOMENT_MODE", "TRAVELING")
    )
    params["STATIC_MOMENT_M0"] = float(params.get("STATIC_MOMENT_M0", 0.0))
    params["INITIAL_BEND_AMPLITUDE"] = float(params.get("INITIAL_BEND_AMPLITUDE", 0.0))
    params["K_SHAPE_MODE"] = normalize_k_shape_mode(params["K_SHAPE_MODE"])
    params["ACTIVE_PHASE_COORDINATE"] = "ACTIVE_BODY_XI"
    params["LAMBDA_ACT_OVER_LACT"] = max(float(params.get("LAMBDA_ACT_OVER_LACT", 1.0)), 1.0e-12)
    params["ACTIVE_PHASE0"] = float(params.get("ACTIVE_PHASE0", 0.0))
    params["WAVE_TIME_SIGN"] = float(params.get("WAVE_TIME_SIGN", 1.0))
    params["C1_S_PASSIVE"] = max(float(params["C1_S_PASSIVE"]), 1.0e-12)
    params["C1_S_PASSIVE_ANTERIOR"] = max(float(params["C1_S_PASSIVE_ANTERIOR"]), 1.0e-12)
    params["C1_S_PASSIVE_PEDUNCLE"] = max(float(params["C1_S_PASSIVE_PEDUNCLE"]), 1.0e-12)
    params["C1_S_PASSIVE_CAUDAL"] = max(float(params["C1_S_PASSIVE_CAUDAL"]), 1.0e-12)
    params["C1_S_PASSIVE_BODY_TRANSITION_S"] = float(
        clamp01(float(params["C1_S_PASSIVE_BODY_TRANSITION_S"]))
    )
    params["C1_S_PASSIVE_BODY_TRANSITION_W"] = max(
        float(params["C1_S_PASSIVE_BODY_TRANSITION_W"]), 1.0e-12
    )
    params["C1_S_PASSIVE_CAUDAL_TRANSITION_S"] = float(
        clamp01(float(params["C1_S_PASSIVE_CAUDAL_TRANSITION_S"]))
    )
    params["C1_S_PASSIVE_CAUDAL_TRANSITION_W"] = max(
        float(params["C1_S_PASSIVE_CAUDAL_TRANSITION_W"]), 1.0e-12
    )
    params["KAPPA_VOL_PASSIVE"] = max(0.0, float(params.get("KAPPA_VOL_PASSIVE", 20.0)))
    params["USE_CONTINUUM_DAMPING"] = parse_bool(params.get("USE_CONTINUUM_DAMPING", False))
    params["CONTINUUM_DAMPING_FACTOR"] = max(
        0.0, float(params.get("CONTINUUM_DAMPING_FACTOR", 0.005))
    )
    params["CONTINUUM_DAMPING_STRESS_CAP_OVER_C1"] = max(
        0.0, float(params.get("CONTINUUM_DAMPING_STRESS_CAP_OVER_C1", 50.0))
    )

    # Populate old generic names for the small amount of plotting code that
    # still treats C1 as an abstract scalar profile.
    params["C1_S"] = params["C1_S_PASSIVE"]
    params["C1_S_ANTERIOR"] = params["C1_S_PASSIVE_ANTERIOR"]
    params["C1_S_PEDUNCLE"] = params["C1_S_PASSIVE_PEDUNCLE"]
    params["C1_S_CAUDAL"] = params["C1_S_PASSIVE_CAUDAL"]
    params["C1_S_BODY_TRANSITION_S"] = params["C1_S_PASSIVE_BODY_TRANSITION_S"]
    params["C1_S_BODY_TRANSITION_W"] = params["C1_S_PASSIVE_BODY_TRANSITION_W"]
    params["C1_S_CAUDAL_TRANSITION_S"] = params["C1_S_PASSIVE_CAUDAL_TRANSITION_S"]
    params["C1_S_CAUDAL_TRANSITION_W"] = params["C1_S_PASSIVE_CAUDAL_TRANSITION_W"]
    params["REFERENCE_PROFILE_BINS"] = max(8, int(params["REFERENCE_PROFILE_BINS"]))
    params["USE_LAPLACE_REFERENCE_PARAMETERIZATION"] = parse_bool(
        params.get("USE_LAPLACE_REFERENCE_PARAMETERIZATION", True)
    )
    params["ALLOW_CENTERLINE_FALLBACK"] = parse_bool(
        params.get("ALLOW_CENTERLINE_FALLBACK", False)
    )
    params["USE_FE_ACTIVE_SECTION_DATA"] = parse_bool(
        params.get("USE_FE_ACTIVE_SECTION_DATA", False)
    )
    params["LAPLACE_HEAD_BC_WIDTH_OVER_L"] = max(
        0.0, float(params.get("LAPLACE_HEAD_BC_WIDTH_OVER_L", 0.05))
    )
    params["LAPLACE_TAIL_BC_WIDTH_OVER_L"] = max(
        0.0, float(params.get("LAPLACE_TAIL_BC_WIDTH_OVER_L", 0.05))
    )
    if params["ACTIVE_S_END"] <= params["ACTIVE_S_START"]:
        raise ValueError("ACTIVE_S_END must be greater than ACTIVE_S_START")
    return params


def infer_mesh_path(input_vals: Dict[str, object], input_path: Path) -> Optional[Path]:
    for key in ("MESH_FILENAME", "FISH_MESH_FILENAME", "MESH_FILE"):
        if key in input_vals:
            p = Path(str(input_vals[key]))
            if not p.is_absolute():
                p = input_path.parent / p
            if p.exists():
                return p.resolve()

    for name in ("fish2d.msh", "fish.msh"):
        p = input_path.parent / name
        if p.exists():
            return p.resolve()

    return None


# ----------------------------------------------------------------------
# Gmsh 2.2 ASCII parser
# ----------------------------------------------------------------------
def parse_gmsh22_ascii(path: Path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    nodes: Dict[int, Tuple[float, float]] = {}
    line_edges: List[Tuple[int, int]] = []
    triangles: List[Tuple[int, int, int]] = []

    while i < len(lines):
        tag = lines[i].strip()
        if tag == "$Nodes":
            n = int(lines[i + 1].strip())
            i += 2
            for _ in range(n):
                parts = lines[i].split()
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                nodes[nid] = (x, y)
                i += 1
            continue

        if tag == "$Elements":
            n = int(lines[i + 1].strip())
            i += 2
            for _ in range(n):
                parts = lines[i].split()
                i += 1
                etype = int(parts[1])
                ntags = int(parts[2])
                conn = list(map(int, parts[3 + ntags:]))
                if etype == 1 and len(conn) >= 2:
                    for a, b in zip(conn[:-1], conn[1:]):
                        line_edges.append((a, b))
                elif etype == 2 and len(conn) >= 3:
                    triangles.append((conn[0], conn[1], conn[2]))
            continue

        i += 1

    if not nodes or not line_edges:
        raise RuntimeError("Failed to parse nodes/line elements from mesh.")
    return nodes, line_edges, triangles


# ----------------------------------------------------------------------
# fish4-4-consistent backbone reconstruction
# ----------------------------------------------------------------------
@dataclass
class CenterlineSegment:
    X0: np.ndarray
    X1: np.ndarray
    t_hat: np.ndarray
    n_hat: np.ndarray
    length: float
    s0: float


@dataclass
class ProjectionResult:
    valid: bool
    seg_id: int
    lam: float
    s: float
    eta: float
    P: np.ndarray
    t_hat: np.ndarray
    n_hat: np.ndarray


@dataclass
class PhiIsoSectionSample:
    valid: bool
    s_norm: float
    s: float
    halfthickness: float
    X_mid: np.ndarray
    t_hat: np.ndarray
    n_hat: np.ndarray


def normalize_reference_vector(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    v_arr = np.asarray(v, dtype=float)
    n = float(np.sqrt(max(float(np.dot(v_arr, v_arr)), 0.0)))
    if n > 1.0e-14:
        return v_arr / n
    f_arr = np.asarray(fallback, dtype=float)
    nf = float(np.sqrt(max(float(np.dot(f_arr, f_arr)), 0.0)))
    if nf > 1.0e-14:
        return f_arr / nf
    return np.array([1.0, 0.0], dtype=float)


def lerp_point(A: np.ndarray, B: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * np.asarray(A, dtype=float) + alpha * np.asarray(B, dtype=float)


def lerp_vector(A: np.ndarray, B: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * np.asarray(A, dtype=float) + alpha * np.asarray(B, dtype=float)


def collect_boundary_edges_from_triangles(triangles: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    edge_order: List[Tuple[int, int]] = []
    for a, b, c in triangles:
        for e0, e1 in ((a, b), (b, c), (c, a)):
            edge = (e0, e1) if e0 <= e1 else (e1, e0)
            if edge_counts[edge] == 0:
                edge_order.append(edge)
            edge_counts[edge] += 1
    return [edge for edge in edge_order if edge_counts[edge] == 1]


def build_polyline_abscissa(pts: List[np.ndarray]) -> np.ndarray:
    s = np.zeros(len(pts), dtype=float)
    for k in range(1, len(pts)):
        s[k] = s[k - 1] + point_distance(pts[k - 1], pts[k])
    return s


def compute_polyline_tangents(pts: List[np.ndarray], s: np.ndarray) -> np.ndarray:
    n = len(pts)
    tang = np.zeros((n, 2), dtype=float)
    if n < 2:
        tang[:, 0] = 1.0
        return tang

    P = np.asarray(pts, dtype=float)
    for k in range(n):
        km = 0 if k == 0 else k - 1
        kp = n - 1 if k + 1 == n else k + 1
        if k == 0:
            kp = 1
        if k + 1 == n:
            km = n - 2
        ds = max(s[kp] - s[km], 1.0e-24)
        tang[k, 0] = (P[kp, 0] - P[km, 0]) / ds
        tang[k, 1] = (P[kp, 1] - P[km, 1]) / ds
    return tang


def cubic_hermite_point(X0, X1, M0, M1, ds, xi):
    h00 = 2.0 * xi**3 - 3.0 * xi**2 + 1.0
    h10 = xi**3 - 2.0 * xi**2 + xi
    h01 = -2.0 * xi**3 + 3.0 * xi**2
    h11 = xi**3 - xi**2
    return np.array([
        h00 * X0[0] + h10 * ds * M0[0] + h01 * X1[0] + h11 * ds * M1[0],
        h00 * X0[1] + h10 * ds * M0[1] + h01 * X1[1] + h11 * ds * M1[1],
    ], dtype=float)


def sample_polyline_cubic_hermite(pts, s, tangents, sq):
    P = np.asarray(pts, dtype=float)
    if len(P) == 0:
        return np.zeros(2)
    if len(P) == 1 or sq <= s[0]:
        return P[0]
    if sq >= s[-1]:
        return P[-1]

    i1 = int(np.searchsorted(s, sq))
    i0 = i1 - 1
    ds = max(s[i1] - s[i0], 1.0e-24)
    xi = (sq - s[i0]) / ds
    return cubic_hermite_point(P[i0], P[i1], tangents[i0], tangents[i1], ds, xi)


def trace_boundary_path(adjacency, start_id, next_id, target_id):
    path = [start_id, next_id]
    prev_id = start_id
    cur_id = next_id
    for _ in range(len(adjacency) + 2):
        if cur_id == target_id:
            return path
        nbrs = adjacency.get(cur_id, [])
        if len(nbrs) != 2:
            return []
        n0, n1 = nbrs
        next_candidate = n1 if n0 == prev_id else n0
        if next_candidate == start_id:
            return []
        path.append(next_candidate)
        prev_id, cur_id = cur_id, next_candidate
    return []


def boundary_path_length(path, points):
    return sum(point_distance(np.array(points[path[k - 1]]), np.array(points[path[k]])) for k in range(1, len(path)))


def extract_shortest_boundary_chain(adjacency, points, start_id, target_id):
    best_path = None
    best_len = 1.0e99
    for next_id in adjacency[start_id]:
        path = trace_boundary_path(adjacency, start_id, next_id, target_id)
        if not path or path[-1] != target_id:
            continue
        L = boundary_path_length(path, points)
        if L < best_len:
            best_len = L
            best_path = path
    if best_path is None:
        raise RuntimeError("Failed to extract shortest boundary chain.")
    return [np.array(points[nid], dtype=float) for nid in best_path]


def truncate_centerline_nodes_at_x(centerline_nodes: List[np.ndarray], x_end: float):
    if len(centerline_nodes) < 2:
        return centerline_nodes

    x_head = float(centerline_nodes[0][0])
    x_tail_full = float(centerline_nodes[-1][0])
    increasing_x = x_tail_full >= x_head
    if increasing_x:
        x_tail = min(max(float(x_end), x_head), x_tail_full)
    else:
        x_tail = max(min(float(x_end), x_head), x_tail_full)

    out = [centerline_nodes[0]]
    for k in range(1, len(centerline_nodes)):
        X0 = centerline_nodes[k - 1]
        X1 = centerline_nodes[k]

        before_tail = (X1[0] < x_tail - 1.0e-12) if increasing_x else (X1[0] > x_tail + 1.0e-12)
        if before_tail:
            out.append(X1)
            continue

        if abs(X1[0] - x_tail) <= 1.0e-12:
            out.append(X1)
        elif (increasing_x and X0[0] < x_tail < X1[0]) or ((not increasing_x) and X1[0] < x_tail < X0[0]):
            dx = X1[0] - X0[0]
            alpha = (x_tail - X0[0]) / dx if abs(dx) > 1.0e-24 else 0.0
            out.append(np.array([x_tail, (1.0 - alpha) * X0[1] + alpha * X1[1]], dtype=float))
        break

    if len(out) < 2:
        raise RuntimeError("Truncated backbone is degenerate.")
    return out


def extend_reference_centerline_to_tail_tip(centerline_nodes: List[np.ndarray],
                                            tail_tip_center_point: np.ndarray,
                                            reference_profile_bins: int,
                                            ref_body_length: float) -> List[np.ndarray]:
    if len(centerline_nodes) < 2:
        return centerline_nodes

    X0 = np.asarray(centerline_nodes[-1], dtype=float)
    X1 = np.asarray(tail_tip_center_point, dtype=float)
    d = point_distance(X0, X1)
    if d <= 1.0e-10 * max(1.0, ref_body_length):
        return centerline_nodes

    n_extra = max(2, int(math.ceil(float(reference_profile_bins) * d / max(ref_body_length, 1.0e-12))))
    out = list(centerline_nodes)
    for k in range(1, n_extra + 1):
        a = k / n_extra
        out.append((1.0 - a) * X0 + a * X1)
    return out


def build_reference_backbone(nodes,
                             line_edges,
                             n_bins: int,
                             requested_backbone_end_x: float,
                             x_leading: float,
                             extend_to_tail_tip: bool = False):
    adjacency = defaultdict(list)
    seen_edges = set()
    for a, b in line_edges:
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        adjacency[a].append(b)
        adjacency[b].append(a)

    for nid, nbrs in adjacency.items():
        if len(nbrs) != 2:
            raise RuntimeError("Boundary is not a simple closed loop; node degree != 2.")

    ref_x_min = min(xy[0] for xy in nodes.values())
    ref_x_max = max(xy[0] for xy in nodes.values())
    ref_body_length = max(ref_x_max - ref_x_min, np.finfo(float).eps)

    dist_to_x_min = abs(float(x_leading) - ref_x_min)
    dist_to_x_max = abs(float(x_leading) - ref_x_max)
    head_at_x_min = dist_to_x_min <= dist_to_x_max
    wave_head_location = "x_min" if head_at_x_min else "x_max"

    if head_at_x_min:
        head_id = min(adjacency.keys(), key=lambda nid: (nodes[nid][0], abs(nodes[nid][1])))
        tail_x = max(nodes[nid][0] for nid in adjacency)
    else:
        head_id = max(adjacency.keys(), key=lambda nid: (nodes[nid][0], -abs(nodes[nid][1])))
        tail_x = min(nodes[nid][0] for nid in adjacency)
    tail_candidates = [nid for nid in adjacency if abs(nodes[nid][0] - tail_x) <= 1.0e-14]
    tail_upper_id = max(tail_candidates, key=lambda nid: nodes[nid][1])
    tail_lower_id = min(tail_candidates, key=lambda nid: nodes[nid][1])
    ref_tail_tip_center_point = np.array(
        [tail_x, 0.5 * (nodes[tail_upper_id][1] + nodes[tail_lower_id][1])],
        dtype=float,
    )

    midline_y_tol = 1.0e-8 * max(1.0, ref_body_length)
    detected_midline_tail_x = None
    for nid in adjacency:
        x, y = nodes[nid]
        if abs(y) <= midline_y_tol:
            if detected_midline_tail_x is None:
                detected_midline_tail_x = x
            else:
                detected_midline_tail_x = max(detected_midline_tail_x, x) if head_at_x_min else min(detected_midline_tail_x, x)

    if math.isfinite(requested_backbone_end_x):
        x_body_lo = min(nodes[head_id][0], tail_x)
        x_body_up = max(nodes[head_id][0], tail_x)
        backbone_end_x = min(max(requested_backbone_end_x, x_body_lo), x_body_up)
    elif detected_midline_tail_x is not None:
        backbone_end_x = detected_midline_tail_x
    else:
        backbone_end_x = tail_x

    upper_chain = extract_shortest_boundary_chain(adjacency, nodes, head_id, tail_upper_id)
    lower_chain = extract_shortest_boundary_chain(adjacency, nodes, head_id, tail_lower_id)

    upper_s = build_polyline_abscissa(upper_chain)
    lower_s = build_polyline_abscissa(lower_chain)
    upper_t = compute_polyline_tangents(upper_chain, upper_s)
    lower_t = compute_polyline_tangents(lower_chain, lower_s)

    n = max(int(n_bins), 8)
    centerline_nodes = []
    for k in range(n):
        xi = k / (n - 1) if n > 1 else 0.0
        Xu = sample_polyline_cubic_hermite(upper_chain, upper_s, upper_t, xi * upper_s[-1])
        Xl = sample_polyline_cubic_hermite(lower_chain, lower_s, lower_t, xi * lower_s[-1])
        Xc = 0.5 * (Xu + Xl)
        centerline_nodes.append(Xc)

    centerline_nodes = truncate_centerline_nodes_at_x(centerline_nodes, backbone_end_x)
    backbone_end_x = float(centerline_nodes[-1][0])
    if extend_to_tail_tip:
        centerline_nodes = extend_reference_centerline_to_tail_tip(
            centerline_nodes, ref_tail_tip_center_point, n, ref_body_length
        )

    ref_profile_s = np.zeros(len(centerline_nodes), dtype=float)
    ref_profile_x = np.array([p[0] for p in centerline_nodes], dtype=float)
    ref_centerline_y = np.array([p[1] for p in centerline_nodes], dtype=float)

    segments: List[CenterlineSegment] = []
    s_accum = 0.0
    for k in range(len(centerline_nodes) - 1):
        X0 = centerline_nodes[k]
        X1 = centerline_nodes[k + 1]
        d = X1 - X0
        length = float(np.linalg.norm(d))
        if length <= 1.0e-12:
            raise RuntimeError("Degenerate backbone segment.")
        t_hat = d / length
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
        segments.append(CenterlineSegment(X0=X0, X1=X1, t_hat=t_hat, n_hat=n_hat, length=length, s0=s_accum))
        s_accum += length
        ref_profile_s[k + 1] = s_accum

    ref_arc_length = max(s_accum, np.finfo(float).eps)

    return {
        "ref_x_min": ref_x_min,
        "ref_x_max": ref_x_max,
        "ref_body_length": ref_body_length,
        "ref_backbone_end_x": backbone_end_x,
        "ref_tail_tip_center_point": ref_tail_tip_center_point,
        "wave_head_location": wave_head_location,
        "head_at_x_min": head_at_x_min,
        "ref_centerline_nodes": centerline_nodes,
        "ref_profile_x": ref_profile_x,
        "ref_centerline_y": ref_centerline_y,
        "ref_profile_s": ref_profile_s,
        "ref_centerline_segments": segments,
        "ref_arc_length": ref_arc_length,
    }


def project_to_reference_centerline(X: np.ndarray, segments: List[CenterlineSegment]) -> ProjectionResult:
    best_r2 = 1.0e99
    best = None
    for i, seg in enumerate(segments):
        d = seg.X1 - seg.X0
        inv_len2 = 1.0 / max(seg.length * seg.length, 1.0e-24)
        lam_raw = float(np.dot(X - seg.X0, d) * inv_len2)
        lam = float(clamp01(lam_raw))
        P = seg.X0 + lam * d
        r = X - P
        r2 = float(np.dot(r, r))
        if r2 + 1.0e-14 < best_r2:
            best_r2 = r2
            eta = float(np.dot(r, seg.n_hat))
            best = ProjectionResult(
                valid=True,
                seg_id=i,
                lam=lam,
                s=seg.s0 + lam * seg.length,
                eta=eta,
                P=P,
                t_hat=seg.t_hat,
                n_hat=seg.n_hat,
            )
    if best is None:
        return ProjectionResult(False, -1, 0.0, 0.0, 0.0, np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    return best


def nearest_reference_profile_index(s: float, profile_s: np.ndarray) -> int:
    if s <= profile_s[0]:
        return 0
    if s >= profile_s[-1]:
        return len(profile_s) - 1
    i1 = int(np.searchsorted(profile_s, s))
    i0 = i1 - 1
    if s - profile_s[i0] <= profile_s[i1] - s:
        return i0
    return i1


def rebuild_reference_halfthickness_from_projection(nodes, geom: Dict[str, object]):
    profile_s = np.asarray(geom["ref_profile_s"], dtype=float)
    n = len(profile_s)
    half = np.zeros(n, dtype=float)
    count = np.zeros(n, dtype=float)
    x_end = float(geom["ref_backbone_end_x"])
    head_at_x_min = bool(geom.get("head_at_x_min", True))
    body_length = float(geom["ref_body_length"])
    segments = geom["ref_centerline_segments"]

    tol = 1.0e-10 * max(1.0, body_length)
    for _, xy in nodes.items():
        X = np.array(xy, dtype=float)
        if head_at_x_min and X[0] > x_end + tol:
            continue
        if (not head_at_x_min) and X[0] < x_end - tol:
            continue
        proj = project_to_reference_centerline(X, segments)
        if not proj.valid:
            continue
        h_sample = max(0.0, abs(proj.eta))
        s_val = float(proj.s)
        if s_val <= float(profile_s[0]):
            half[0] = max(half[0], h_sample)
            count[0] += 1.0
        elif s_val >= float(profile_s[-1]):
            half[-1] = max(half[-1], h_sample)
            count[-1] += 1.0
        else:
            i1 = int(np.searchsorted(profile_s, s_val, side='right'))
            i0 = i1 - 1
            half[i0] = max(half[i0], h_sample)
            half[i1] = max(half[i1], h_sample)
            count[i0] += 1.0
            count[i1] += 1.0

    if np.all(count <= 0.5):
        raise RuntimeError("No occupied s bins in half-thickness reconstruction.")

    for k in range(n):
        if count[k] > 0.5:
            continue
        kl = k - 1
        while kl >= 0 and count[kl] <= 0.5:
            kl -= 1
        kr = k + 1
        while kr < n and count[kr] <= 0.5:
            kr += 1
        if kl >= 0 and kr < n:
            alpha = (profile_s[k] - profile_s[kl]) / max(profile_s[kr] - profile_s[kl], 1.0e-24)
            half[k] = (1.0 - alpha) * half[kl] + alpha * half[kr]
        elif kl >= 0:
            half[k] = half[kl]
        elif kr < n:
            half[k] = half[kr]

    half_raw = half.copy()

    if n > 2:
        for _ in range(3):
            smoothed = half.copy()
            smoothed[0] = max(float(smoothed[0]), float(half_raw[0]))
            smoothed[-1] = max(float(smoothed[-1]), float(half_raw[-1]))
            interior = 0.25 * half[:-2] + 0.50 * half[1:-1] + 0.25 * half[2:]
            smoothed[1:-1] = np.maximum(interior, half_raw[1:-1])
            half = smoothed

        half = np.maximum(half, half_raw)

    h_max = float(np.max(half))
    if h_max <= 1.0e-12:
        raise RuntimeError("Degenerate h(s) after reconstruction.")

    geom["ref_halfthickness_raw"] = half_raw
    geom["ref_halfthickness"] = half
    geom["ref_halfthickness_norm"] = half / h_max
    geom["ref_h_max"] = h_max
    return geom


def triangle_area_and_grads(coords: np.ndarray) -> Tuple[float, np.ndarray]:
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    x2, y2 = coords[2]
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    area = 0.5 * abs(det)
    if area <= 1.0e-30:
        raise RuntimeError("Degenerate triangle in Laplace assembly.")
    grads = np.array([
        [y1 - y2, x2 - x1],
        [y2 - y0, x0 - x2],
        [y0 - y1, x1 - x0],
    ], dtype=float) / det
    return area, grads


def solve_laplace_phi_p1(nodes: Dict[int, Tuple[float, float]],
                         triangles: List[Tuple[int, int, int]],
                         head_patch_node_ids: set,
                         tail_patch_node_ids: set) -> Dict[int, float]:
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except Exception as exc:
        raise RuntimeError(
            "Strict Laplace reference parameterization requires scipy.sparse. "
            "Install SciPy or set USE_LAPLACE_REFERENCE_PARAMETERIZATION = FALSE."
        ) from exc

    node_ids = sorted(nodes.keys())
    node_to_i = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    K = sp.lil_matrix((n, n), dtype=float)

    for tri in triangles:
        ids = list(tri)
        coords = np.array([nodes[nid] for nid in ids], dtype=float)
        area, grads = triangle_area_and_grads(coords)
        Ke = area * (grads @ grads.T)
        for a_local, a_id in enumerate(ids):
            ia = node_to_i[a_id]
            for b_local, b_id in enumerate(ids):
                K[ia, node_to_i[b_id]] += Ke[a_local, b_local]

    constrained_values: Dict[int, float] = {}
    for nid in head_patch_node_ids:
        constrained_values[nid] = 0.0
    for nid in tail_patch_node_ids:
        if nid in constrained_values:
            raise RuntimeError("Laplace head/tail boundary node sets overlap.")
        constrained_values[nid] = 1.0

    constrained = np.array(sorted(node_to_i[nid] for nid in constrained_values), dtype=int)
    if constrained.size == 0:
        raise RuntimeError("No constrained nodes for Laplace solve.")
    free_mask = np.ones(n, dtype=bool)
    free_mask[constrained] = False
    free = np.nonzero(free_mask)[0]

    phi = np.zeros(n, dtype=float)
    for nid, value in constrained_values.items():
        phi[node_to_i[nid]] = value

    K = K.tocsr()
    if free.size > 0:
        rhs = -K[free][:, constrained].dot(phi[constrained])
        phi[free] = spla.spsolve(K[free][:, free], rhs)

    return {nid: float(phi[node_to_i[nid]]) for nid in node_ids}


def interpolate_phi_section_sample(sections: List[PhiIsoSectionSample],
                                   s_norm_query: float) -> PhiIsoSectionSample:
    if not sections:
        raise RuntimeError("Phi isocontour section table is empty.")
    q = float(clamp01(s_norm_query))
    if q <= sections[0].s_norm:
        return sections[0]
    if q >= sections[-1].s_norm:
        return sections[-1]

    s_vals = np.array([sec.s_norm for sec in sections], dtype=float)
    i1 = int(np.searchsorted(s_vals, q, side="right"))
    i0 = i1 - 1
    A = sections[i0]
    B = sections[i1]
    alpha = float(clamp01((q - A.s_norm) / max(B.s_norm - A.s_norm, 1.0e-24)))
    t_hat = normalize_reference_vector(lerp_vector(A.t_hat, B.t_hat, alpha), A.t_hat)
    n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
    if float(np.dot(n_hat, A.n_hat)) < 0.0:
        n_hat *= -1.0
    return PhiIsoSectionSample(
        valid=A.valid and B.valid,
        s_norm=q,
        s=(1.0 - alpha) * A.s + alpha * B.s,
        halfthickness=max(0.0, (1.0 - alpha) * A.halfthickness + alpha * B.halfthickness),
        X_mid=lerp_point(A.X_mid, B.X_mid, alpha),
        t_hat=t_hat,
        n_hat=n_hat,
    )


def build_phi_isocontour_section_table(nodes: Dict[int, Tuple[float, float]],
                                       boundary_edges: List[Tuple[int, int]],
                                       phi_node: Dict[int, float],
                                       grad_phi_node: Dict[int, np.ndarray],
                                       geom: Dict[str, object],
                                       reference_profile_bins: int) -> List[PhiIsoSectionSample]:
    n_sections = max(int(reference_profile_bins), 8)
    sections: List[PhiIsoSectionSample] = []
    phi_tol = 1.0e-12
    dup_tol2 = 1.0e-20 * max(float(geom["ref_body_length"]) * float(geom["ref_body_length"]), 1.0)
    fallback_t = np.array([1.0, 0.0] if bool(geom.get("head_at_x_min", True)) else [-1.0, 0.0], dtype=float)

    for k in range(n_sections):
        s_norm = k / (n_sections - 1) if n_sections > 1 else 0.0
        intersections: List[Tuple[np.ndarray, np.ndarray]] = []

        for id_a, id_b in boundary_edges:
            if id_a not in phi_node or id_b not in phi_node:
                continue
            phi_a = float(phi_node[id_a])
            phi_b = float(phi_node[id_b])
            if s_norm < min(phi_a, phi_b) - phi_tol or s_norm > max(phi_a, phi_b) + phi_tol:
                continue

            if abs(phi_b - phi_a) > phi_tol:
                alpha = float(clamp01((s_norm - phi_a) / (phi_b - phi_a)))
            else:
                if abs(s_norm - phi_a) > phi_tol:
                    continue
                alpha = 0.5

            A = np.array(nodes[id_a], dtype=float)
            B = np.array(nodes[id_b], dtype=float)
            X = lerp_point(A, B, alpha)
            ga = grad_phi_node.get(id_a, fallback_t)
            gb = grad_phi_node.get(id_b, fallback_t)
            grad = normalize_reference_vector(lerp_vector(ga, gb, alpha), np.array([1.0, 0.0], dtype=float))

            duplicate = False
            for X_prev, _ in intersections:
                d = X - X_prev
                if float(np.dot(d, d)) <= dup_tol2:
                    duplicate = True
                    break
            if not duplicate:
                intersections.append((X, grad))

        section = PhiIsoSectionSample(
            valid=False,
            s_norm=s_norm,
            s=0.0,
            halfthickness=0.0,
            X_mid=np.array([0.0, 0.0], dtype=float),
            t_hat=np.array([1.0, 0.0], dtype=float),
            n_hat=np.array([0.0, 1.0], dtype=float),
        )

        if len(intersections) >= 2:
            t_sum = np.sum([grad for _, grad in intersections], axis=0)
            t_hat = normalize_reference_vector(t_sum, fallback_t)
            n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
            p_vals = [float(np.dot(X, n_hat)) for X, _ in intersections]
            i_min = int(np.argmin(p_vals))
            i_max = int(np.argmax(p_vals))
            p_min = p_vals[i_min]
            p_max = p_vals[i_max]
            X_min = intersections[i_min][0]
            X_max = intersections[i_max][0]
            h = 0.5 * max(p_max - p_min, 0.0)
            if h > 1.0e-12:
                section = PhiIsoSectionSample(
                    valid=True,
                    s_norm=s_norm,
                    s=0.0,
                    halfthickness=h,
                    X_mid=0.5 * (X_min + X_max),
                    t_hat=t_hat,
                    n_hat=n_hat,
                )
        sections.append(section)

    valid_indices = [i for i, sec in enumerate(sections) if sec.valid]
    if not valid_indices:
        raise RuntimeError("No valid phi-isocontour boundary sections.")
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]

    for k in range(first_valid - 1, -1, -1):
        src = sections[first_valid]
        sections[k] = PhiIsoSectionSample(
            valid=src.valid,
            s_norm=k / (n_sections - 1),
            s=src.s,
            halfthickness=src.halfthickness,
            X_mid=src.X_mid.copy(),
            t_hat=src.t_hat.copy(),
            n_hat=src.n_hat.copy(),
        )
    for k in range(last_valid + 1, n_sections):
        src = sections[last_valid]
        sections[k] = PhiIsoSectionSample(
            valid=src.valid,
            s_norm=k / (n_sections - 1),
            s=src.s,
            halfthickness=src.halfthickness,
            X_mid=src.X_mid.copy(),
            t_hat=src.t_hat.copy(),
            n_hat=src.n_hat.copy(),
        )

    for k in range(first_valid, last_valid + 1):
        if sections[k].valid:
            continue
        kl = k - 1
        while kl >= first_valid and not sections[kl].valid:
            kl -= 1
        kr = k + 1
        while kr <= last_valid and not sections[kr].valid:
            kr += 1
        if kl < first_valid or kr > last_valid:
            raise RuntimeError("Failed to fill invalid phi-section.")
        s_norm_k = k / (n_sections - 1)
        alpha = float(clamp01((s_norm_k - sections[kl].s_norm) /
                              max(sections[kr].s_norm - sections[kl].s_norm, 1.0e-24)))
        t_hat = normalize_reference_vector(lerp_vector(sections[kl].t_hat, sections[kr].t_hat, alpha),
                                           sections[kl].t_hat)
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
        if float(np.dot(n_hat, sections[kl].n_hat)) < 0.0:
            n_hat *= -1.0
        sections[k] = PhiIsoSectionSample(
            valid=True,
            s_norm=s_norm_k,
            s=0.0,
            halfthickness=(1.0 - alpha) * sections[kl].halfthickness + alpha * sections[kr].halfthickness,
            X_mid=lerp_point(sections[kl].X_mid, sections[kr].X_mid, alpha),
            t_hat=t_hat,
            n_hat=n_hat,
        )

    midline_length = 0.0
    for k in range(n_sections - 1):
        midline_length += point_distance(sections[k].X_mid, sections[k + 1].X_mid)
    midline_length = max(midline_length, max(float(geom["ref_body_length"]), 1.0e-12))

    for k, sec in enumerate(sections):
        s_norm = k / (n_sections - 1) if n_sections > 1 else 0.0
        sec.valid = True
        sec.s_norm = s_norm
        sec.s = s_norm * midline_length
    return sections


def rebuild_reference_centerline_from_phi_sections(geom: Dict[str, object],
                                                   sections: List[PhiIsoSectionSample]) -> Dict[str, object]:
    if len(sections) < 2:
        raise RuntimeError("Need at least two phi sections.")
    ref_centerline_nodes = [sec.X_mid.copy() for sec in sections]
    ref_profile_x = np.array([sec.X_mid[0] for sec in sections], dtype=float)
    ref_centerline_y = np.array([sec.X_mid[1] for sec in sections], dtype=float)
    ref_profile_s = np.array([sec.s for sec in sections], dtype=float)
    ref_halfthickness = np.array([sec.halfthickness for sec in sections], dtype=float)
    ref_arc_length = max(float(ref_profile_s[-1] - ref_profile_s[0]), 1.0e-12)
    ref_h_max = float(np.max(ref_halfthickness))
    if ref_h_max <= 1.0e-12:
        raise RuntimeError("h(phi) is degenerate.")

    segments: List[CenterlineSegment] = []
    for k in range(len(sections) - 1):
        X0 = sections[k].X_mid
        X1 = sections[k + 1].X_mid
        seg_len = max(sections[k + 1].s - sections[k].s, 1.0e-12)
        t_hat = normalize_reference_vector(
            lerp_vector(sections[k].t_hat, sections[k + 1].t_hat, 0.5),
            X1 - X0,
        )
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
        segments.append(CenterlineSegment(X0=X0, X1=X1, t_hat=t_hat, n_hat=n_hat, length=seg_len, s0=sections[k].s))

    geom = dict(geom)
    geom["ref_phi_sections"] = sections
    geom["ref_centerline_nodes"] = ref_centerline_nodes
    geom["ref_profile_x"] = ref_profile_x
    geom["ref_centerline_y"] = ref_centerline_y
    geom["ref_profile_s"] = ref_profile_s
    geom["ref_halfthickness"] = ref_halfthickness
    geom["ref_halfthickness_raw"] = ref_halfthickness.copy()
    geom["ref_halfthickness_norm"] = ref_halfthickness / ref_h_max
    geom["ref_centerline_segments"] = segments
    geom["ref_arc_length"] = ref_arc_length
    geom["ref_h_max"] = ref_h_max
    return geom


def build_reference_laplace_parameterization(nodes: Dict[int, Tuple[float, float]],
                                             triangles: List[Tuple[int, int, int]],
                                             geom: Dict[str, object],
                                             params: Dict[str, object]) -> Dict[str, object]:
    if not triangles:
        raise RuntimeError("Strict Laplace reference parameterization requires triangular volume elements.")

    ref_x_min = float(geom["ref_x_min"])
    ref_x_max = float(geom["ref_x_max"])
    Lx = max(ref_x_max - ref_x_min, 1.0e-12)
    head_at_x_min = bool(geom.get("head_at_x_min", True))
    head_width = max(0.0, float(params["LAPLACE_HEAD_BC_WIDTH_OVER_L"])) * Lx
    tail_width = max(0.0, float(params["LAPLACE_TAIL_BC_WIDTH_OVER_L"])) * Lx

    boundary_edges = collect_boundary_edges_from_triangles(triangles)
    if not boundary_edges:
        raise RuntimeError("No boundary edges found from triangular elements.")

    boundary_node_ids = set()
    head_patch_node_ids = set()
    tail_patch_node_ids = set()
    head_side_count = 0
    tail_side_count = 0
    for a, b in boundary_edges:
        boundary_node_ids.update((a, b))
        x_centroid = 0.5 * (nodes[a][0] + nodes[b][0])
        in_head_patch = (
            x_centroid <= ref_x_min + head_width + 1.0e-12
            if head_at_x_min else
            x_centroid >= ref_x_max - head_width - 1.0e-12
        )
        in_tail_patch = (
            x_centroid >= ref_x_max - tail_width - 1.0e-12
            if head_at_x_min else
            x_centroid <= ref_x_min + tail_width + 1.0e-12
        )
        if in_head_patch and in_tail_patch:
            raise RuntimeError("Laplace head/tail boundary patches overlap.")
        if in_head_patch:
            head_side_count += 1
            head_patch_node_ids.update((a, b))
        if in_tail_patch:
            tail_side_count += 1
            tail_patch_node_ids.update((a, b))

    if head_side_count < 1 or tail_side_count < 1 or not head_patch_node_ids or not tail_patch_node_ids:
        raise RuntimeError(
            "Failed to mark finite head/tail Laplace boundary side patches. "
            f"head sides/nodes = {head_side_count}/{len(head_patch_node_ids)}, "
            f"tail sides/nodes = {tail_side_count}/{len(tail_patch_node_ids)}"
        )

    raw_phi_node = solve_laplace_phi_p1(nodes, triangles, head_patch_node_ids, tail_patch_node_ids)
    raw_vals = np.array(list(raw_phi_node.values()), dtype=float)
    raw_min = float(np.min(raw_vals))
    raw_max = float(np.max(raw_vals))
    phi_node = {nid: float(clamp01(val)) for nid, val in raw_phi_node.items()}
    phi_vals = np.array(list(phi_node.values()), dtype=float)
    phi_min = float(np.min(phi_vals))
    phi_max = float(np.max(phi_vals))
    head_bc_error = max(abs(raw_phi_node[nid]) for nid in head_patch_node_ids)
    tail_bc_error = max(abs(raw_phi_node[nid] - 1.0) for nid in tail_patch_node_ids)

    if head_bc_error > 1.0e-10 or tail_bc_error > 1.0e-10:
        raise RuntimeError(
            "Laplace Dirichlet constraints were not enforced exactly. "
            f"max |phi(head)-0| = {head_bc_error}, max |phi(tail)-1| = {tail_bc_error}"
        )
    if raw_min < -1.0e-10 or raw_max > 1.0 + 1.0e-10:
        raise RuntimeError(f"Constrained Laplace coordinate left [0,1]: range = [{raw_min}, {raw_max}]")
    if not (phi_min <= 0.05 and phi_max >= 0.95 and phi_max - phi_min >= 0.50):
        raise RuntimeError(
            "Degenerate harmonic coordinate range "
            f"[{phi_min}, {phi_max}], constrained range [{raw_min}, {raw_max}]"
        )

    grad_sum: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=float))
    grad_weight: Dict[int, float] = defaultdict(float)
    fallback_t = np.array([1.0, 0.0] if head_at_x_min else [-1.0, 0.0], dtype=float)
    for tri in triangles:
        ids = list(tri)
        coords = np.array([nodes[nid] for nid in ids], dtype=float)
        area, grads = triangle_area_and_grads(coords)
        grad_elem = np.zeros(2, dtype=float)
        for local, nid in enumerate(ids):
            grad_elem += phi_node[nid] * grads[local]
        for nid in ids:
            grad_sum[nid] += area * grad_elem
            grad_weight[nid] += area

    grad_phi_node: Dict[int, np.ndarray] = {}
    for nid in nodes:
        g = fallback_t.copy()
        if grad_weight[nid] > 0.0:
            g = grad_sum[nid] / grad_weight[nid]
        grad_phi_node[nid] = normalize_reference_vector(g, fallback_t)

    x_cut = float(geom["ref_backbone_end_x"])
    min_dx = min(abs(nodes[nid][0] - x_cut) for nid in boundary_node_ids)
    cut_tol = min_dx + 1.0e-8 * max(1.0, Lx)
    cut_phi = [phi_node[nid] for nid in boundary_node_ids if abs(nodes[nid][0] - x_cut) <= cut_tol]
    active_end_s_norm = float(clamp01(np.mean(cut_phi))) if cut_phi else approximate_s_norm_from_x(x_cut, geom)

    sections = build_phi_isocontour_section_table(
        nodes=nodes,
        boundary_edges=boundary_edges,
        phi_node=phi_node,
        grad_phi_node=grad_phi_node,
        geom=geom,
        reference_profile_bins=int(params["REFERENCE_PROFILE_BINS"]),
    )
    geom = rebuild_reference_centerline_from_phi_sections(geom, sections)
    geom["laplace_phi_node"] = phi_node
    geom["laplace_grad_phi_node"] = grad_phi_node
    geom["active_end_s_norm"] = active_end_s_norm
    geom["laplace_head_side_count"] = head_side_count
    geom["laplace_tail_side_count"] = tail_side_count
    geom["laplace_head_node_count"] = len(head_patch_node_ids)
    geom["laplace_tail_node_count"] = len(tail_patch_node_ids)
    geom["laplace_raw_phi_range"] = (raw_min, raw_max)
    geom["laplace_phi_range"] = (phi_min, phi_max)
    geom["reference_parameterization"] = "LAPLACE_GRAPH_HARMONIC_STRICT_P1"
    return geom


# ----------------------------------------------------------------------
# fish4-4 formulas
# ----------------------------------------------------------------------
def wave_ramp(time: np.ndarray, wave_ramp_time: float) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    if wave_ramp_time <= 0.0:
        return np.ones_like(time)

    R = np.zeros_like(time)
    mid = (time > 0.0) & (time < wave_ramp_time)
    R[mid] = 0.5 * (1.0 - np.cos(np.pi * time[mid] / wave_ramp_time))
    R[time >= wave_ramp_time] = 1.0
    return R


def longitudinal_active_envelope(
    s_norm: np.ndarray,
    active_s_start: float,
    active_s_end: float,
    active_s_smooth: float,
):
    s_norm = clamp01(np.asarray(s_norm, dtype=float))
    w = np.zeros_like(s_norm)

    inside = (s_norm > active_s_start) & (s_norm < active_s_end)
    w[inside] = 1.0

    span = max(active_s_end - active_s_start, 0.0)
    if span <= 1.0e-12:
        return w

    ds = max(min(active_s_smooth, 0.5 * span), 1.0e-12)
    left = inside & (s_norm < active_s_start + ds)
    w[left] *= smoothstep((s_norm[left] - active_s_start) / ds)

    right = inside & (s_norm > active_s_end - ds)
    w[right] *= smoothstep((active_s_end - s_norm[right]) / ds)
    return w


def apply_effective_actuation_params(params: Dict[str, object], geom: Dict[str, object]) -> Dict[str, object]:
    Lref = max(float(geom["ref_arc_length"]), 1.0e-12)
    params = dict(params)
    s0 = active_s_start_norm_effective(params)
    s1 = active_s_end_norm_effective(params, geom)
    span = max(s1 - s0, 0.0)
    params["ACTIVE_S_START_EFFECTIVE"] = s0
    params["ACTIVE_S_END_EFFECTIVE"] = s1
    params["ACTIVE_S_SPAN_EFFECTIVE"] = span
    params["ACTIVE_PHASE_LENGTH"] = max(span * Lref, 1.0e-12)
    params["ACTIVE_PHASE_COORDINATE"] = "ACTIVE_BODY_XI"
    params["ACTIVE_PHASE_WAVELENGTH"] = (
        float(params["LAMBDA_ACT_OVER_LACT"]) * float(params["ACTIVE_PHASE_LENGTH"])
    )
    wave_time_sign = float(params.get("WAVE_TIME_SIGN", 1.0))
    ds_dt_sign = 1.0 if wave_time_sign > 0.0 else (-1.0 if wave_time_sign < 0.0 else 0.0)
    dx_ds_sign = 1.0 if bool(geom.get("head_at_x_min", True)) else -1.0
    params["WAVE_PROPAGATION_X_SIGN"] = ds_dt_sign * dx_ds_sign
    params["WAVE_PROPAGATION_S"] = (
        "toward increasing phase coordinate (head-to-tail)" if wave_time_sign > 0.0 else
        "toward decreasing phase coordinate (tail-to-head)" if wave_time_sign < 0.0 else
        "standing wave in phase coordinate"
    )
    return params


def reference_x_norm_from_s(s: np.ndarray, geom: Dict[str, object]) -> np.ndarray:
    s_arr = np.asarray(s, dtype=float)
    ref_s = np.asarray(geom["ref_profile_s"], dtype=float)
    ref_x = np.asarray(geom["ref_profile_x"], dtype=float)
    x = np.interp(np.clip(s_arr, ref_s[0], ref_s[-1]), ref_s, ref_x)
    Lx = max(float(geom.get("fish_length", geom["ref_body_length"])), 1.0e-12)
    x_leading = float(geom.get("x_leading", geom["ref_x_min"] if bool(geom.get("head_at_x_min", True)) else geom["ref_x_max"]))
    if bool(geom.get("head_at_x_min", True)):
        return clamp01((x - x_leading) / Lx)
    return clamp01((x_leading - x) / Lx)


def active_moment_shape_from_s(s: np.ndarray, params: Dict[str, object], geom: Dict[str, object]) -> np.ndarray:
    s_arr = np.asarray(s, dtype=float)
    ref_arc_length = max(float(geom["ref_arc_length"]), 1.0e-12)
    s_norm = clamp01(s_arr / ref_arc_length)
    xi_active = active_xi_from_s_norm(s_norm, params, geom)
    mode = str(params["K_SHAPE_MODE"]).upper()

    if mode == "HALF-BELL":
        return 0.5 * (1.0 - np.cos(np.pi * xi_active))

    if mode == "BELL":
        return 1.0 - np.cos(2.0 * np.pi * xi_active)

    raise ValueError(f'K_SHAPE_MODE must be "HALF-BELL" or "BELL", got {mode!r}')


def approximate_s_norm_from_x(x_query: float, geom: Dict[str, object]) -> float:
    ref_x = np.asarray(geom["ref_profile_x"], dtype=float)
    ref_s = np.asarray(geom["ref_profile_s"], dtype=float)
    ref_arc_length = max(float(geom["ref_arc_length"]), 1.0e-12)
    if len(ref_x) >= 2 and len(ref_s) == len(ref_x):
        order = np.argsort(ref_x)
        x_sorted = ref_x[order]
        s_sorted = ref_s[order]
        xq = float(np.clip(x_query, x_sorted[0], x_sorted[-1]))
        return float(clamp01(np.interp(xq, x_sorted, s_sorted) / ref_arc_length))
    Lx = max(float(geom["ref_x_max"]) - float(geom["ref_x_min"]), 1.0e-12)
    if bool(geom.get("head_at_x_min", True)):
        return float(clamp01((x_query - float(geom["ref_x_min"])) / Lx))
    return float(clamp01((float(geom["ref_x_max"]) - x_query) / Lx))


def active_end_s_norm_effective(params: Dict[str, object], geom: Dict[str, object]) -> float:
    cached = float(geom.get("active_end_s_norm", float("nan")))
    if math.isfinite(cached):
        return float(clamp01(cached))
    return approximate_s_norm_from_x(float(geom["ref_backbone_end_x"]), geom)


def active_s_start_norm_effective(params: Dict[str, object]) -> float:
    return float(clamp01(float(params.get("ACTIVE_S_START", 0.0))))


def active_s_end_norm_effective(params: Dict[str, object], geom: Dict[str, object]) -> float:
    return float(clamp01(float(params.get("ACTIVE_S_END", 1.0))))


def active_xi_from_s_norm(s_norm: np.ndarray, params: Dict[str, object], geom: Dict[str, object]) -> np.ndarray:
    s = clamp01(np.asarray(s_norm, dtype=float))
    s0 = active_s_start_norm_effective(params)
    s1 = active_s_end_norm_effective(params, geom)
    span = max(s1 - s0, 0.0)
    if span <= 1.0e-12:
        return np.zeros_like(s, dtype=float)
    return clamp01((s - s0) / span)


def active_moment_drive_amplitude_from_s(s: np.ndarray,
                                         params: Dict[str, object],
                                         geom: Dict[str, object]) -> np.ndarray:
    s_arr = np.asarray(s, dtype=float)
    return np.abs(active_moment_shape_from_s(s_arr, params, geom))


def active_phase_angle_from_s(s: np.ndarray,
                              t: np.ndarray,
                              params: Dict[str, object],
                              geom: Dict[str, object]) -> np.ndarray:
    s_arr = np.asarray(s, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    omega = 2.0 * np.pi * float(params["WAVE_FREQUENCY"])
    wave_time_sign = float(params.get("WAVE_TIME_SIGN", 1.0))
    s_norm = clamp01(s_arr / max(float(geom["ref_arc_length"]), 1.0e-12))
    xi = active_xi_from_s_norm(s_norm, params, geom)
    spatial_phase = 2.0 * np.pi * xi / max(float(params["LAMBDA_ACT_OVER_LACT"]), 1.0e-12)
    return spatial_phase[None, :] - wave_time_sign * omega * t_arr[:, None] + float(params["ACTIVE_PHASE0"])


def active_moment_drive_from_s_time(s: np.ndarray,
                                    t: np.ndarray,
                                    params: Dict[str, object],
                                    geom: Dict[str, object]) -> np.ndarray:
    s_arr = np.asarray(s, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    phase = active_phase_angle_from_s(s_arr, t_arr, params, geom)
    drive_amp_shape = active_moment_shape_from_s(s_arr, params, geom)
    return drive_amp_shape[None, :] * np.cos(phase)


def c1_s_profile_from_coords(s_norm: np.ndarray, x_norm: np.ndarray, params: Dict[str, object]) -> np.ndarray:
    x_arr = clamp01(np.asarray(x_norm, dtype=float))
    w_body = smoothstep_cosine(
        x_arr,
        float(params["C1_S_BODY_TRANSITION_S"]),
        float(params["C1_S_BODY_TRANSITION_W"]),
    )
    w_caudal = smoothstep_cosine(
        x_arr,
        float(params["C1_S_CAUDAL_TRANSITION_S"]),
        float(params["C1_S_CAUDAL_TRANSITION_W"]),
    )
    return (
        float(params["C1_S_ANTERIOR"]) +
        (float(params["C1_S_PEDUNCLE"]) - float(params["C1_S_ANTERIOR"])) * w_body +
        (float(params["C1_S_CAUDAL"]) - float(params["C1_S_PEDUNCLE"])) * w_caudal
    )


def build_static_profiles(params: Dict[str, object], geom: Dict[str, object]):
    s = np.asarray(geom["ref_profile_s"], dtype=float)
    ref_arc_length = float(geom["ref_arc_length"])
    h = np.asarray(geom["ref_halfthickness"], dtype=float)
    h_raw = np.asarray(geom.get("ref_halfthickness_raw", h), dtype=float)
    s_norm = s / max(ref_arc_length, 1.0e-12)
    active_xi = active_xi_from_s_norm(s_norm, params, geom)
    x_norm = reference_x_norm_from_s(s, geom)

    K_shape = active_moment_shape_from_s(s, params, geom)
    drive_amp = active_moment_drive_amplitude_from_s(s, params, geom)
    h_scale = np.maximum(h, 0.0) ** 2
    K_s = float(params["BETA_ACT"]) * h_scale * drive_amp
    I2_unit = active_band_second_moment_unit(float(params["ACTIVE_BAND_FRACTION"]))
    i2_h_power = float(params["ACTIVE_I2_H_POWER"])
    h_power = np.maximum(h, 1.0e-12) ** i2_h_power
    I2 = I2_unit * h_power
    I2_floor = float(params["FE_SECTION_I2_FLOOR_RATIO"]) * I2
    # The preprocessor usually has only the reference geometry, not the FE
    # quadrature table. Runtime C++ uses max(I2_c_FE_scaled, I2_floor) when
    # FE section data are enabled; here I2_use_proxy records the analytic
    # ideal used for pre-run plots and becomes the floor reference for
    # diagnosing FE I2_c.
    I2_use_proxy = I2.copy()
    K_over_h_power = np.divide(K_s, h_power, out=np.zeros_like(K_s), where=h_power > 0.0)
    active_s0 = float(params.get("ACTIVE_S_START_EFFECTIVE", active_s_start_norm_effective(params)))
    active_s1 = float(params.get("ACTIVE_S_END_EFFECTIVE", active_s_end_norm_effective(params, geom)))
    env_s = longitudinal_active_envelope(
        s_norm=s_norm,
        active_s_start=active_s0,
        active_s_end=active_s1,
        active_s_smooth=float(params["ACTIVE_S_SMOOTH"]),
    )
    active_mask = ((s_norm > active_s0) & (s_norm < active_s1)).astype(float)
    if uses_static_active_moment(params):
        env_K = active_mask * float(params["STATIC_MOMENT_M0"])
    else:
        env_K = env_s * K_s
    env_K_over_h_power = np.divide(env_K, h_power, out=np.zeros_like(env_K), where=h_power > 0.0)
    env_K_over_I2 = np.divide(env_K, I2_use_proxy, out=np.zeros_like(env_K), where=I2_use_proxy > 0.0)
    c1_s_profile = c1_s_profile_from_coords(s_norm, x_norm, params)
    T_act_max = float(params["ACTIVE_T_ACT_MAX_OVER_C1"]) * c1_s_profile
    Mm_max = np.divide(T_act_max * I2_use_proxy, h, out=np.zeros_like(I2_use_proxy), where=h > 1.0e-30)
    stress_sign = float(params["ACTIVE_MOMENT_TO_STRESS_SIGN"])
    Tact_upper_proxy = np.divide(
        stress_sign * env_K * h,
        I2_use_proxy,
        out=np.zeros_like(env_K),
        where=I2_use_proxy > 1.0e-30,
    )
    Tact_lower_proxy = -Tact_upper_proxy
    Mm_clamped_proxy = np.zeros_like(env_K)
    valid_cap = Mm_max > 1.0e-30
    Mm_clamped_proxy[valid_cap] = (
        Mm_max[valid_cap] * np.tanh(env_K[valid_cap] / Mm_max[valid_cap])
    )
    Tact_upper_clamped_proxy = np.divide(
        stress_sign * Mm_clamped_proxy * h,
        I2_use_proxy,
        out=np.zeros_like(env_K),
        where=I2_use_proxy > 1.0e-30,
    )
    Tact_lower_clamped_proxy = -Tact_upper_clamped_proxy
    Tact_abs_clamped_proxy = np.maximum(
        np.abs(Tact_upper_clamped_proxy), np.abs(Tact_lower_clamped_proxy)
    )

    return {
        "s": s,
        "s_norm": s_norm,
        "active_xi": active_xi,
        "x_norm": x_norm,
        "h_raw": h_raw,
        "h": h,
        "K_shape": K_shape,
        "drive_amp": drive_amp,
        "I2": I2,
        "I2_floor": I2_floor,
        "I2_use_proxy": I2_use_proxy,
        "I2_eff_unit": np.full_like(s_norm, I2_unit, dtype=float),
        "h_scale": h_scale,
        "K_s": K_s,
        "K_over_h3": K_over_h_power,
        "active_i2_h_power": np.full_like(s_norm, i2_h_power, dtype=float),
        "env_s": env_s,
        "env_K": env_K,
        "env_K_over_h3": env_K_over_h_power,
        "env_K_over_I2": env_K_over_I2,
        "C1_S": c1_s_profile,
        "Mm_max": Mm_max,
        "Tact_upper_proxy": Tact_upper_proxy,
        "Tact_lower_proxy": Tact_lower_proxy,
        "Tact_upper_clamped_proxy": Tact_upper_clamped_proxy,
        "Tact_lower_clamped_proxy": Tact_lower_clamped_proxy,
        "Tact_abs_clamped_proxy": Tact_abs_clamped_proxy,
        "ref_arc_length": ref_arc_length,
    }


def nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(arr - value)))


def selected_Mm_profile_indices(ts: Dict[str, object], max_profiles: int = 5) -> np.ndarray:
    t = np.asarray(ts["t"], dtype=float)
    if t.size <= 0:
        return np.asarray([], dtype=int)
    n = min(max_profiles, t.size)
    return np.unique(np.linspace(0, t.size - 1, n, dtype=int))


def time_column_tag(t_value: float) -> str:
    return f"t_{t_value:.6g}".replace("-", "m").replace("+", "").replace(".", "p")


def build_time_series(params: Dict[str, object], static: Dict[str, np.ndarray], t_start: float, t_end: float, nt: int,
                      s_fracs: List[float]):
    s = np.asarray(static["s"], dtype=float)
    env_s = np.asarray(static["env_s"], dtype=float)

    t = np.linspace(t_start, t_end, nt)
    R = wave_ramp(t, float(params["WAVE_RAMP_TIME"]))
    omega = 2.0 * np.pi * float(params["WAVE_FREQUENCY"])
    coord = np.asarray(static["active_xi"], dtype=float)
    spatial_phase = 2.0 * np.pi * coord / max(float(params["LAMBDA_ACT_OVER_LACT"]), 1.0e-12)
    phase_t0 = spatial_phase + float(params["ACTIVE_PHASE0"])
    phase = (spatial_phase[None, :] -
             float(params.get("WAVE_TIME_SIGN", 1.0)) * omega * t[:, None] +
             float(params["ACTIVE_PHASE0"]))
    drive = np.asarray(static["drive_amp"], dtype=float)[None, :] * np.cos(phase)
    drive_t0 = np.asarray(static["drive_amp"], dtype=float) * np.cos(phase_t0)

    drive_amp = np.asarray(static["drive_amp"], dtype=float)
    phase_carrier = np.divide(drive, drive_amp[None, :],
                              out=np.zeros_like(drive),
                              where=drive_amp[None, :] > 1.0e-30)
    phase_carrier_t0 = np.divide(drive_t0, drive_amp,
                                 out=np.zeros_like(drive_amp),
                                 where=drive_amp > 1.0e-30)
    if uses_static_active_moment(params):
        active_s0 = float(params["ACTIVE_S_START_EFFECTIVE"])
        active_s1 = float(params["ACTIVE_S_END_EFFECTIVE"])
        active_mask = (
            (np.asarray(static["s_norm"], dtype=float) > active_s0) &
            (np.asarray(static["s_norm"], dtype=float) < active_s1)
        ).astype(float)
        Mm_raw = R[:, None] * float(params["STATIC_MOMENT_M0"]) * active_mask[None, :]
        phase_carrier = np.broadcast_to(active_mask[None, :], Mm_raw.shape).copy()
        phase_carrier_t0 = active_mask.copy()
    elif float(params["BETA_ACT"]) <= 0.0:
        Mm_raw = np.zeros_like(drive)
    else:
        Mm_raw = R[:, None] * env_s[None, :] * (
            float(params["BETA_ACT"]) * np.asarray(static["h_scale"], dtype=float)
        )[None, :] * drive
    Mm_max = np.asarray(static["Mm_max"], dtype=float)
    Mm_clamped = np.zeros_like(Mm_raw)
    valid_clamp = Mm_max > 1.0e-30
    if np.any(valid_clamp):
        Mm_scale = Mm_max[valid_clamp][None, :]
        Mm_clamped[:, valid_clamp] = Mm_scale * np.tanh(Mm_raw[:, valid_clamp] / Mm_scale)
    stress_sign = float(params["ACTIVE_MOMENT_TO_STRESS_SIGN"])
    stress_section_moment = stress_sign * Mm_clamped
    curvature_conjugate_moment = -stress_sign * Mm_clamped
    Mm_abs = np.abs(Mm_raw)
    Mm_rms = np.sqrt(np.mean(Mm_raw * Mm_raw, axis=0))
    Mm_clamped_rms = np.sqrt(np.mean(Mm_clamped * Mm_clamped, axis=0))

    overlay_scale = np.max(static["env_K"]) if np.max(static["env_K"]) > 0.0 else 1.0
    phase_carrier_overlay = 0.5 * (phase_carrier + 1.0) * overlay_scale

    points = []
    for frac in s_fracs:
        idx = nearest_index(static["s_norm"], float(frac))
        points.append({
            "s_frac_req": float(frac),
            "idx": idx,
            "s": float(s[idx]),
            "s_norm": float(static["s_norm"][idx]),
            "Mm_raw_t": Mm_raw[:, idx].copy(),
            "Mm_clamped_t": Mm_clamped[:, idx].copy(),
            "curvature_moment_t": curvature_conjugate_moment[:, idx].copy(),
            "Mm_abs_t": Mm_abs[:, idx].copy(),
        })

    return {
        "t": t,
        "R": R,
        "omega": omega,
        "phase_carrier": phase_carrier,
        "phase_carrier_t0": phase_carrier_t0,
        "phase_carrier_overlay": phase_carrier_overlay,
        "Mm_raw": Mm_raw,
        "Mm_clamped": Mm_clamped,
        "stress_section_moment": stress_section_moment,
        "curvature_conjugate_moment": curvature_conjugate_moment,
        "Mm_abs": Mm_abs,
        "Mm_rms": Mm_rms,
        "Mm_clamped_rms": Mm_clamped_rms,
        "points": points,
    }


# ----------------------------------------------------------------------
# csv and plots
# ----------------------------------------------------------------------
def save_csvs(out_dir: Path, static: Dict[str, np.ndarray], ts: Dict[str, object]):
    out_dir.mkdir(parents=True, exist_ok=True)

    static_cols = np.column_stack([
        static["s"],
        static["s_norm"],
        static["active_xi"],
        static["x_norm"],
        static["h_raw"],
        static["h"],
        static["K_shape"],
        static["drive_amp"],
        static["I2"],
        static["I2_floor"],
        static["I2_use_proxy"],
        static["I2_eff_unit"],
        static["active_i2_h_power"],
        static["h_scale"],
        static["K_s"],
        static["K_over_h3"],
        static["env_s"],
        static["env_K"],
        static["env_K_over_h3"],
        static["env_K_over_I2"],
        static["C1_S"],
        static["Mm_max"],
        static["Tact_upper_proxy"],
        static["Tact_lower_proxy"],
        static["Tact_upper_clamped_proxy"],
        static["Tact_lower_clamped_proxy"],
        static["Tact_abs_clamped_proxy"],
    ])
    np.savetxt(
        out_dir / "actuation_static_profile.csv",
        static_cols,
        delimiter=",",
        header="s,s_norm,active_xi,x_norm,h_raw,h,K_shape,drive_amp,I2_ideal,I2_floor,I2_use_proxy,I2_eff_unit,ACTIVE_I2_H_POWER,h_scale,K_s,K_over_h_power,env_s,env_K,env_K_over_h_power,env_K_over_I2,C1_S_PASSIVE,Mm_max,Tact_upper_proxy,Tact_lower_proxy,Tact_upper_clamped_proxy,Tact_lower_clamped_proxy,Tact_abs_clamped_proxy",
        comments="",
    )

    cols = [np.asarray(ts["t"]), np.asarray(ts["R"])]
    header = ["time", "R"]
    for p in ts["points"]:
        tag = f"sfrac_{p['s_norm']:.3f}"
        cols.append(p["Mm_raw_t"])
        cols.append(p["Mm_clamped_t"])
        cols.append(p["curvature_moment_t"])
        cols.append(p["Mm_abs_t"])
        header.append(f"Mm_raw_{tag}")
        header.append(f"Mm_clamped_{tag}")
        header.append(f"curvature_moment_{tag}")
        header.append(f"Mm_abs_{tag}")
    np.savetxt(
        out_dir / "actuation_time_series.csv",
        np.column_stack(cols),
        delimiter=",",
        header=",".join(header),
        comments="",
    )

    np.savetxt(
        out_dir / "actuation_Mm_rms_profile.csv",
        np.column_stack([
            static["s_norm"],
            static["active_xi"],
            static["x_norm"],
            ts["Mm_rms"],
            ts["Mm_clamped_rms"],
            static["env_K"],
            static["Mm_max"],
        ]),
        delimiter=",",
        header="s_norm,active_xi,x_norm,Mm_rms,Mm_clamped_rms,env_K,Mm_max",
        comments="",
    )

    profile_indices = selected_Mm_profile_indices(ts)
    if profile_indices.size > 0:
        profile_cols = [
            np.asarray(static["s_norm"], dtype=float),
            np.asarray(static.get("active_xi", np.full_like(static["s_norm"], np.nan)), dtype=float),
            np.asarray(static["x_norm"], dtype=float),
        ]
        profile_header = ["s_norm", "active_xi", "x_norm"]
        for idx in profile_indices:
            t_value = float(np.asarray(ts["t"], dtype=float)[idx])
            tag = time_column_tag(t_value)
            profile_cols.append(np.asarray(ts["Mm_raw"], dtype=float)[idx, :])
            profile_cols.append(np.asarray(ts["Mm_clamped"], dtype=float)[idx, :])
            profile_cols.append(np.asarray(ts["Mm_abs"], dtype=float)[idx, :])
            profile_header.append(f"Mm_raw_{tag}")
            profile_header.append(f"Mm_clamped_{tag}")
            profile_header.append(f"Mm_abs_{tag}")
        np.savetxt(
            out_dir / "actuation_Mm_profiles_s.csv",
            np.column_stack(profile_cols),
            delimiter=",",
            header=",".join(profile_header),
            comments="",
        )


def plot_Ks_only(out_dir: Path, static: Dict[str, np.ndarray], active_s_start: float, active_s_end: float):
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    ax.plot(static["s_norm"], static["K_s"], linewidth=2.2,
            label="K_s = beta*h(s)^2*K_shape(xi)")
    ax.axvline(active_s_start, linestyle=":", linewidth=1.2, label="ACTIVE_S_START")
    ax.axvline(active_s_end, linestyle=":", linewidth=1.2, label="ACTIVE_S_END")
    ax.set_xlabel("s / L_backbone (0=head, 1=tail)")
    ax.set_ylabel("active moment amplitude")
    ax.set_title("Active moment envelope K_s (mesh-consistent)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "00_Ks_only.png", bbox_inches="tight")
    plt.close(fig)


def plot_static(out_dir: Path, static: Dict[str, np.ndarray], active_s_start: float, active_s_end: float):
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    ax.plot(static["s_norm"], static["K_s"], linewidth=2.0,
            label="K_s = beta*h(s)^2*K_shape(xi)")
    ax.plot(static["s_norm"], static["env_K"], linewidth=2.0,
            label="env_s*K_s")
    ax.plot(static["s_norm"], static["env_s"], linewidth=1.8, label="env_s")
    ax.axvline(active_s_start, linestyle=":", linewidth=1.2, label="ACTIVE_S_START")
    ax.axvline(active_s_end, linestyle=":", linewidth=1.2, label="ACTIVE_S_END")
    ax.set_xlabel("s / L_backbone (0=head, 1=tail)")
    ax.set_ylabel("amplitude")
    ax.set_title("Static moment-envelope profiles (mesh-consistent)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "01_static_profiles.png", bbox_inches="tight")
    plt.close(fig)


def plot_ramp(out_dir: Path, ts: Dict[str, object], wave_ramp_time: float):
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    ax.plot(ts["t"], ts["R"], linewidth=2.2, label="R(t) = wave_ramp(time)")
    ax.axvline(wave_ramp_time, linestyle=":", linewidth=1.2, label="WAVE_RAMP_TIME")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("R(t)")
    ax.set_title("Ramp function over time")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "02_ramp_vs_time.png", bbox_inches="tight")
    plt.close(fig)


def plot_point_timeseries(out_dir: Path, ts: Dict[str, object]):
    points = ts["points"]
    if not points:
        return

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)

    for p in points:
        ax.plot(
            ts["t"],
            p["Mm_raw_t"],
            linewidth=2.0,
            label=f"Mm_raw, s/L≈{p['s_norm']:.2f}",
        )
        ax.plot(
            ts["t"],
            p["Mm_clamped_t"],
            linewidth=1.5,
            linestyle="--",
            label=f"Mm_clamped, s/L≈{p['s_norm']:.2f}",
        )

    ax.set_xlabel("time [s]")
    ax.set_ylabel("moment")
    ax.set_title("Raw and tanh-saturated active moment at selected positions")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "03_Mm_time_series_points.png", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(out_dir: Path, static: Dict[str, np.ndarray], ts: Dict[str, object]):
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=180)
    im = ax.imshow(
        ts["Mm_raw"],
        aspect="auto",
        origin="lower",
        extent=[static["s_norm"][0], static["s_norm"][-1], ts["t"][0], ts["t"][-1]],
    )
    ax.set_xlabel("s / L_backbone")
    ax.set_ylabel("time [s]")
    ax.set_title("Mm_raw(s,t) heatmap (mesh-consistent)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mm_raw")
    fig.tight_layout()
    fig.savefig(out_dir / "04_Mm_raw_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_Mm_rms_profile(out_dir: Path, static: Dict[str, np.ndarray], ts: Dict[str, object],
                        active_s_start: float, active_s_end: float):
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    ax.plot(static["s_norm"], ts["Mm_rms"], linewidth=2.2, label="Mm_rms(s)")
    ax.plot(static["s_norm"], ts["Mm_clamped_rms"], linewidth=1.8,
            linestyle="--", label="Mm_clamped_rms(s)")
    env = np.asarray(static["env_K"], dtype=float)
    rms = np.asarray(ts["Mm_rms"], dtype=float)
    if np.max(env) > 0.0 and np.max(rms) > 0.0:
        ax.plot(static["s_norm"], env / np.max(env) * np.max(rms),
                linewidth=1.6, linestyle="--", label="K_eff(s), scaled")
    ax.axvline(active_s_start, linestyle=":", linewidth=1.2, label="ACTIVE_S_START")
    ax.axvline(active_s_end, linestyle=":", linewidth=1.2, label="ACTIVE_S_END")
    ax.set_xlabel("s / L_backbone")
    ax.set_ylabel("moment rms")
    ax.set_title("Active moment RMS profile")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "08_Mm_rms_profile.png", bbox_inches="tight")
    plt.close(fig)


def plot_Mm_s_profiles(out_dir: Path, static: Dict[str, np.ndarray], ts: Dict[str, object],
                       active_s_start: float, active_s_end: float):
    profile_indices = selected_Mm_profile_indices(ts)
    if profile_indices.size <= 0:
        return

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    t = np.asarray(ts["t"], dtype=float)
    Mm_raw = np.asarray(ts["Mm_raw"], dtype=float)
    Mm_clamped = np.asarray(ts["Mm_clamped"], dtype=float)
    for idx in profile_indices:
        ax.plot(static["s_norm"], Mm_raw[idx, :], linewidth=1.9, label=f"t={t[idx]:.3g}s")
        ax.plot(static["s_norm"], Mm_clamped[idx, :], linewidth=1.2,
                linestyle="--", alpha=0.85, label=f"clamped t={t[idx]:.3g}s")
    ax.axhline(0.0, color="k", linestyle=":", linewidth=1.0, alpha=0.65)
    ax.axvline(active_s_start, linestyle=":", linewidth=1.2, label="ACTIVE_S_START")
    ax.axvline(active_s_end, linestyle=":", linewidth=1.2, label="ACTIVE_S_END")
    ax.set_xlabel("s / L_backbone")
    ax.set_ylabel("moment")
    ax.set_title("Raw and tanh-saturated active moment profiles along s/L")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "09_Mm_profiles_s.png", bbox_inches="tight")
    plt.close(fig)


def plot_C1S_profile(out_dir: Path, static: Dict[str, np.ndarray], params: Dict[str, object]):
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    coord = np.asarray(static["s_norm"], dtype=float)
    c1s_plot = np.asarray(static["C1_S"], dtype=float)
    coord_label = "s / L_backbone"

    def xnorm_to_snorm(xi_value: float) -> float:
        x_arr = np.asarray(static["x_norm"], dtype=float)
        s_arr = np.asarray(static["s_norm"], dtype=float)
        order = np.argsort(x_arr)
        return float(np.interp(float(clamp01(xi_value)), x_arr[order], s_arr[order]))

    ax.plot(coord, c1s_plot, linewidth=2.2, label="C1_S_PASSIVE")
    for label, center_key, width_key in [
        ("body transition", "C1_S_BODY_TRANSITION_S", "C1_S_BODY_TRANSITION_W"),
        ("caudal transition", "C1_S_CAUDAL_TRANSITION_S", "C1_S_CAUDAL_TRANSITION_W"),
    ]:
        center_xi = float(params[center_key])
        width = float(params[width_key])
        cp = xnorm_to_snorm(center_xi)
        lo = xnorm_to_snorm(center_xi - 0.5 * width)
        hi = xnorm_to_snorm(center_xi + 0.5 * width)
        ax.axvline(cp, linestyle=":", linewidth=1.2, label=label)
        ax.axvspan(max(0.0, min(lo, hi)), min(1.0, max(lo, hi)), alpha=0.10)
    ax.set_xlabel(coord_label)
    ax.set_ylabel("C1_S_PASSIVE")
    ax.set_title("Spatial passive continuum stiffness profile")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "05_C1S_profile.png", bbox_inches="tight")
    plt.close(fig)


def plot_K_over_h3(out_dir: Path, static: Dict[str, np.ndarray],
                   active_s_start: float, active_s_end: float):
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    p = float(np.asarray(static.get("active_i2_h_power", [3.0]), dtype=float)[0])
    ax.plot(static["s_norm"], static["K_over_h3"], linewidth=2.2,
            label=f"K_s/h(s)^{p:g}")
    ax.plot(static["s_norm"], static["env_K_over_h3"], linewidth=1.8,
            label=f"env_s*K_s/h(s)^{p:g}")
    ax.axvline(active_s_start, linestyle=":", linewidth=1.2, label="ACTIVE_S_START")
    ax.axvline(active_s_end, linestyle=":", linewidth=1.2, label="ACTIVE_S_END")
    ax.set_xlabel("s / L_backbone (0=head, 1=tail)")
    ax.set_ylabel("moment/I2 proxy")
    ax.set_title("Active stress proxy and boundary T_act")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(static["s_norm"], static["Tact_upper_clamped_proxy"],
             linewidth=1.7, linestyle="--", color="tab:red",
             label="T_act upper, clamped proxy")
    ax2.plot(static["s_norm"], static["Tact_lower_clamped_proxy"],
             linewidth=1.7, linestyle=":", color="tab:blue",
             label="T_act lower, clamped proxy")
    ax2.set_ylabel("T_act at eta=+/-h")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "06_K_over_h3.png", bbox_inches="tight")
    plt.close(fig)


def plot_phase_carrier(out_dir: Path, static: Dict[str, np.ndarray], ts: Dict[str, object]):
    fig, axs = plt.subplots(2, 1, figsize=(9.2, 7.0), dpi=180, sharex=False)

    ax = axs[0]
    ax.plot(static["s_norm"], ts["phase_carrier_t0"], linewidth=2.0,
            label="cos(phase), t=0")
    ax.axhline(0.0, color="k", linestyle=":", linewidth=1.0, alpha=0.65)
    ax.set_xlabel("s / L_backbone")
    ax.set_ylabel("phase carrier")
    ax.set_title("Active phase carrier at t=0")
    ax.set_ylim(-1.08, 1.08)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axs[1]
    im = ax.imshow(
        ts["phase_carrier"],
        aspect="auto",
        origin="lower",
        vmin=-1.0,
        vmax=1.0,
        extent=[static["s_norm"][0], static["s_norm"][-1], ts["t"][0], ts["t"][-1]],
    )
    ax.set_xlabel("s / L_backbone")
    ax.set_ylabel("time [s]")
    ax.set_title("phase_carrier(s,t)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("drive / drive_amp")

    fig.tight_layout()
    fig.savefig(out_dir / "07_phase_carrier.png", bbox_inches="tight")
    plt.close(fig)


def plot_overview(out_dir: Path, static: Dict[str, np.ndarray], ts: Dict[str, object],
                  active_s_start: float, active_s_end: float, wave_ramp_time: float):
    fig, axs = plt.subplots(2, 2, figsize=(12.0, 8.5), dpi=180)

    ax = axs[0, 0]
    ax.plot(static["s_norm"], static["K_s"], linewidth=2.0, label="K(s)")
    ax.plot(static["s_norm"], static["env_K"], linewidth=2.0, label="active_mask * env_s * K(s)")
    ax.axvline(active_s_start, linestyle=":", linewidth=1.2)
    ax.axvline(active_s_end, linestyle=":", linewidth=1.2)
    ax.set_xlabel("s / L_backbone")
    ax.set_ylabel("amplitude")
    ax.set_title("Static profiles")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axs[0, 1]
    ax.plot(ts["t"], ts["R"], linewidth=2.0, label="R(t)")
    ax.axvline(wave_ramp_time, linestyle=":", linewidth=1.2, label="WAVE_RAMP_TIME")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("R(t)")
    ax.set_title("Ramp")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axs[1, 0]
    for p in ts["points"]:
        ax.plot(ts["t"], p["Mm_raw_t"], linewidth=1.5, label=f"Mm_raw, s/L≈{p['s_norm']:.2f}")
        ax.plot(ts["t"], p["Mm_clamped_t"], linewidth=1.2, linestyle="--",
                label=f"Mm_clamped, s/L≈{p['s_norm']:.2f}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("moment")
    ax.set_title("Raw/clamped moment at selected positions")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axs[1, 1]
    im = ax.imshow(
        ts["Mm_raw"],
        aspect="auto",
        origin="lower",
        extent=[static["s_norm"][0], static["s_norm"][-1], ts["t"][0], ts["t"][-1]],
    )
    ax.set_xlabel("s / L_backbone")
    ax.set_ylabel("time [s]")
    ax.set_title("Mm_raw(s,t)")
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", bbox_inches="tight")
    plt.close(fig)


def write_readme(out_dir: Path, params: Dict[str, object], geom: Dict[str, object], args):
    k_mode = str(params["K_SHAPE_MODE"]).upper()
    h_description = (
        "strict phi-isocontour boundary-section half-thickness, matching C++ Laplace mode"
        if "LAPLACE" in str(geom.get("reference_parameterization", "")).upper()
        else "centerline-projection half-thickness profile with the C++ smoothing/envelope fallback"
    )
    if k_mode == "HALF-BELL":
        k_shape_description = "half-bell 0.5*(1-cos(pi*xi)) [posterior-rising, 0 to 1 head-to-tail]"
    else:
        k_shape_description = "bell 1-cos(2*pi*xi) [Xu/Zhou/Yu 2024 Eq. (2), zero at head and tail]"
    moment_mode = normalize_active_moment_mode(params.get("ACTIVE_MOMENT_MODE", "TRAVELING"))
    if moment_mode == "STATIC":
        moment_formula = "Mm_raw = R(t)*STATIC_MOMENT_M0 inside ACTIVE_S_START < s_norm < ACTIVE_S_END"
        k_s_description = "K_s remains the traveling-mode beta_act*h(s)^2*K_shape(xi) reference; env_K is the static moment profile."
        drive_description = "phase and K_shape are diagnostic only in STATIC mode."
    else:
        moment_formula = "Mm_raw = R(t)*env_s(s)*beta_act*h(s)^2*K_shape(xi)*cos(phase)"
        k_s_description = "K_s is beta_act*h(s)^2*K_shape(xi), matching current C++ TRAVELING mode."
        drive_description = "drive_amp is K_shape(xi); old FACTORIZED caudal gain and h-power terms are not used by current C++."
    laplace_extra = ""
    if "LAPLACE" in str(geom.get("reference_parameterization", "")).upper():
        laplace_extra = f"""
- LAPLACE_HEAD_BC_WIDTH_OVER_L = {params['LAPLACE_HEAD_BC_WIDTH_OVER_L']}
- LAPLACE_TAIL_BC_WIDTH_OVER_L = {params['LAPLACE_TAIL_BC_WIDTH_OVER_L']}
- Laplace head sides/nodes = {geom.get('laplace_head_side_count', 'n/a')} / {geom.get('laplace_head_node_count', 'n/a')}
- Laplace tail sides/nodes = {geom.get('laplace_tail_side_count', 'n/a')} / {geom.get('laplace_tail_node_count', 'n/a')}
- Laplace raw phi range = {geom.get('laplace_raw_phi_range', 'n/a')}
- mapped reference-end s_norm = {geom.get('active_end_s_norm', 'n/a')}
"""

    txt = f"""Generated files
- 00_Ks_only.png
- 01_static_profiles.png
- 02_ramp_vs_time.png
- 03_Mm_time_series_points.png
- 04_Mm_raw_heatmap.png
- 05_C1S_profile.png
- 06_K_over_h3.png
- 07_phase_carrier.png
- 08_Mm_rms_profile.png
- 09_Mm_profiles_s.png
- overview.png
- actuation_static_profile.csv
- actuation_time_series.csv
- actuation_Mm_rms_profile.csv
- actuation_Mm_profiles_s.csv

Interpretation
- ACTIVE_MOMENT_MODE = {moment_mode}.
- {moment_formula}.
- {k_s_description}
- env_K is the unit-ramp active moment envelope used for plotting: env_s*K_s in TRAVELING mode, static M0 on the active interval in STATIC mode.
- K_shape mode = {params['K_SHAPE_MODE']}: {k_shape_description}.
- {drive_description}
- h(s) is the final C++ half-thickness profile: {h_description}.
- C1_S_PASSIVE(s) is the passive continuum stiffness profile used by raw deviatoric PK1 stress and by the active stress cap.
- K_over_h_power = K_s/h(s)^ACTIVE_I2_H_POWER. This is the active stress-amplitude scale before eta, phase sign, ramp, band weight, and clamp; its peak can differ from K_s because h(s) is in the denominator.
- env_K_over_h_power = env_K/h(s)^ACTIVE_I2_H_POWER, including the longitudinal active envelope.
- I2_ideal = I2_eff_unit*h(s)^ACTIVE_I2_H_POWER; I2_floor = FE_SECTION_I2_FLOOR_RATIO*I2_ideal.
- Runtime C++ uses I2_use=max(I2_c_FE_scaled,I2_floor) when FE section data are enabled. I2_c_FE_scaled rescales the raw FE I2_c by I2_ideal/(I2_eff_unit*h^3), so ACTIVE_I2_H_POWER still affects the active stress with FE section data enabled.
- env_K_over_I2 = env_K/I2_use_proxy, the pre-run analytic stress scale corresponding to the current I2 floor convention.
- Mm_max = ACTIVE_T_ACT_MAX_OVER_C1*C1_S_PASSIVE*I2_use_proxy/h and Mm_clamped = Mm_max*tanh(Mm_raw/Mm_max), matching the current C++ smooth saturation up to the FE I2_c correction.
- Tact_upper/lower_proxy = ACTIVE_MOMENT_TO_STRESS_SIGN*env_K*(+/-h)/I2_use_proxy. The clamped columns use Mm_max*tanh(env_K/Mm_max). These are pre-run eta_bar=0 proxies for the C++ T_act at the upper/lower boundary.
- 07_phase_carrier.png shows cos(phase) at t=0 and phase_carrier(s,t), so LAMBDA_ACT_OVER_LACT changes are visible even when static amplitude envelopes are unchanged.
- R(t), Mm_raw(s,t), Mm_clamped(s,t), and Mm_abs(s,t) are time-dependent.
- 09_Mm_profiles_s.png and actuation_Mm_profiles_s.csv preserve raw and clamped profiles versus s/L at representative times.
- This run used time window: {args.t_start} ~ {args.t_end} s
- selected s/L positions = {args.s_frac}

Parameters
- FISH_LENGTH = {params['FISH_LENGTH']}
- WAVE_FREQUENCY = {params['WAVE_FREQUENCY']}
- WAVE_RAMP_TIME = {params['WAVE_RAMP_TIME']}
- BETA_ACT = {params['BETA_ACT']}
- ACTIVE_MOMENT_MODE = {params['ACTIVE_MOMENT_MODE']}
- STATIC_MOMENT_M0 = {params['STATIC_MOMENT_M0']}
- INITIAL_BEND_AMPLITUDE = {params['INITIAL_BEND_AMPLITUDE']}
- ACTIVE_PHASE_COORDINATE = {params.get('ACTIVE_PHASE_COORDINATE', 'ACTIVE_BODY_XI')}
- LAMBDA_ACT_OVER_LACT = {params.get('LAMBDA_ACT_OVER_LACT', 1.0)}
- ACTIVE_PHASE0 = {params.get('ACTIVE_PHASE0', 0.0)}
- ACTIVE_S_START_EFFECTIVE = {params.get('ACTIVE_S_START_EFFECTIVE', params['ACTIVE_S_START'])}
- ACTIVE_S_END_EFFECTIVE = {params.get('ACTIVE_S_END_EFFECTIVE', params['ACTIVE_S_END'])} (from ACTIVE_S_END, not capped by mapped reference-end s_norm)
- ACTIVE_PHASE_LENGTH = {params.get('ACTIVE_PHASE_LENGTH', 'n/a')}
- active phase wavelength = {params.get('ACTIVE_PHASE_WAVELENGTH', 'n/a')}
- WAVE_TIME_SIGN = {params.get('WAVE_TIME_SIGN', 1.0)}
- active drive phase convention = cos(k*xi - WAVE_TIME_SIGN*omega*t + ACTIVE_PHASE0)
- active phase-speed direction = {params.get('WAVE_PROPAGATION_S', 'toward increasing phase coordinate (head-to-tail)')}
- C1_S_PASSIVE = {params['C1_S_PASSIVE']}
- C1_S_PASSIVE_ANTERIOR = {params['C1_S_PASSIVE_ANTERIOR']}
- C1_S_PASSIVE_PEDUNCLE = {params['C1_S_PASSIVE_PEDUNCLE']}
- C1_S_PASSIVE_CAUDAL = {params['C1_S_PASSIVE_CAUDAL']}
- C1_S_PASSIVE_BODY_TRANSITION_S = {params['C1_S_PASSIVE_BODY_TRANSITION_S']}
- C1_S_PASSIVE_BODY_TRANSITION_W = {params['C1_S_PASSIVE_BODY_TRANSITION_W']}
- C1_S_PASSIVE_CAUDAL_TRANSITION_S = {params['C1_S_PASSIVE_CAUDAL_TRANSITION_S']}
- C1_S_PASSIVE_CAUDAL_TRANSITION_W = {params['C1_S_PASSIVE_CAUDAL_TRANSITION_W']}
- KAPPA_VOL_PASSIVE = {params['KAPPA_VOL_PASSIVE']}
- USE_CONTINUUM_DAMPING = {params['USE_CONTINUUM_DAMPING']}
- CONTINUUM_DAMPING_FACTOR = {params['CONTINUUM_DAMPING_FACTOR']}
- CONTINUUM_DAMPING_STRESS_CAP_OVER_C1 = {params['CONTINUUM_DAMPING_STRESS_CAP_OVER_C1']}
- ACTIVE_S_START = {params['ACTIVE_S_START']}
- ACTIVE_S_END = {params['ACTIVE_S_END']}
- ACTIVE_S_SMOOTH = {params['ACTIVE_S_SMOOTH']}
- ACTIVE_I2_H_POWER = {params['ACTIVE_I2_H_POWER']}
- ACTIVE_MOMENT_TO_STRESS_SIGN = {params['ACTIVE_MOMENT_TO_STRESS_SIGN']}
- ACTIVE_T_ACT_MAX_OVER_C1 = {params['ACTIVE_T_ACT_MAX_OVER_C1']}
- FE_SECTION_I2_FLOOR_RATIO = {params['FE_SECTION_I2_FLOOR_RATIO']}
- ACTIVE_BAND_FRACTION = {params['ACTIVE_BAND_FRACTION']}
- K_SHAPE_MODE = {params['K_SHAPE_MODE']}
- REFERENCE_PROFILE_BINS = {params['REFERENCE_PROFILE_BINS']}
- REFERENCE_BACKBONE_END_X = {params['REFERENCE_BACKBONE_END_X']}
- USE_LAPLACE_REFERENCE_PARAMETERIZATION = {params['USE_LAPLACE_REFERENCE_PARAMETERIZATION']}
- ALLOW_CENTERLINE_FALLBACK = {params['ALLOW_CENTERLINE_FALLBACK']}
- USE_FE_ACTIVE_SECTION_DATA = {params['USE_FE_ACTIVE_SECTION_DATA']}

Geometry summary
- reference_parameterization = {geom.get('reference_parameterization', 'UNKNOWN')}
- ref_x_min = {geom['ref_x_min']}
- ref_x_max = {geom['ref_x_max']}
- ref_body_length = {geom['ref_body_length']}
- wave_head_location = {geom.get('wave_head_location', 'x_min')}
- active phase-speed x sign = {params.get('WAVE_PROPAGATION_X_SIGN', 0.0)}
- ref_backbone_end_x = {geom['ref_backbone_end_x']}
- ref_arc_length = {geom['ref_arc_length']}
- ref_h_max = {geom['ref_h_max']}
{laplace_extra.rstrip()}
"""
    (out_dir / "README.txt").write_text(txt, encoding="utf-8")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Mesh-consistent time-window actuation plots using fish4-4 input + cpp model.")
    p.add_argument("input2d", type=str, help="Path to input2d")
    p.add_argument("--mesh", type=str, default=None, help="Path to fish2d.msh / fish.msh (optional if inferable)")
    p.add_argument("--cpp", type=str, default=None, help="Optional cpp for parsing fallback defaults")
    p.add_argument("--t-start", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=4.0)
    p.add_argument("--nt", type=int, default=700)
    p.add_argument("--s-frac", type=float, nargs="*", default=[0.50, 0.70, 0.85])
    p.add_argument("--output-dir", type=str, default="figs_time_mesh_consistent")
    return p.parse_args()


def main():
    args = parse_args()

    if args.t_end <= args.t_start:
        raise ValueError("t_end must be greater than t_start")

    input_path = Path(args.input2d).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input2d not found: {input_path}")

    input_vals = parse_input2d(input_path)
    cpp_vals = {}
    if args.cpp:
        cpp_path = Path(args.cpp).expanduser().resolve()
        if cpp_path.exists():
            cpp_vals = parse_cpp_defaults(cpp_path)

    params = build_param_dict(input_vals, cpp_vals)

    mesh_path = Path(args.mesh).expanduser().resolve() if args.mesh else infer_mesh_path(input_vals, input_path)
    if mesh_path is None or not mesh_path.exists():
        raise FileNotFoundError("Mesh file not found. Provide --mesh or set MESH_FILENAME in input2d.")

    nodes, line_edges, triangles = parse_gmsh22_ascii(mesh_path)
    use_laplace_reference = bool(params["USE_LAPLACE_REFERENCE_PARAMETERIZATION"])
    geom = build_reference_backbone(
        nodes=nodes,
        line_edges=line_edges,
        n_bins=int(params["REFERENCE_PROFILE_BINS"]),
        requested_backbone_end_x=float(params["REFERENCE_BACKBONE_END_X"]),
        x_leading=float(params["X_LEADING"]),
        extend_to_tail_tip=use_laplace_reference,
    )
    geom["x_leading"] = float(params["X_LEADING"])
    geom["fish_length"] = float(params["FISH_LENGTH"])
    if use_laplace_reference:
        try:
            geom = build_reference_laplace_parameterization(nodes, triangles, geom, params)
        except Exception:
            if not bool(params["ALLOW_CENTERLINE_FALLBACK"]):
                raise
            print("[WARN] strict Laplace parameterization failed; falling back to centerline projection.")
            geom = rebuild_reference_halfthickness_from_projection(nodes, geom)
            geom["reference_parameterization"] = "CENTERLINE_PROJECTION_FALLBACK"
    else:
        geom = rebuild_reference_halfthickness_from_projection(nodes, geom)
        geom["reference_parameterization"] = "CENTERLINE_PROJECTION"
    params = apply_effective_actuation_params(params, geom)

    static = build_static_profiles(params, geom)
    ts = build_time_series(
        params=params,
        static=static,
        t_start=args.t_start,
        t_end=args.t_end,
        nt=args.nt,
        s_fracs=args.s_frac,
    )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_csvs(out_dir, static, ts)
    active_s0 = float(params.get("ACTIVE_S_START_EFFECTIVE", params["ACTIVE_S_START"]))
    active_s1 = float(params.get("ACTIVE_S_END_EFFECTIVE", params["ACTIVE_S_END"]))
    plot_Ks_only(out_dir, static, active_s0, active_s1)
    plot_static(out_dir, static, active_s0, active_s1)
    plot_ramp(out_dir, ts, float(params["WAVE_RAMP_TIME"]))
    plot_point_timeseries(out_dir, ts)
    plot_heatmap(out_dir, static, ts)
    plot_C1S_profile(out_dir, static, params)
    plot_K_over_h3(out_dir, static, active_s0, active_s1)
    plot_phase_carrier(out_dir, static, ts)
    plot_Mm_rms_profile(out_dir, static, ts, active_s0, active_s1)
    plot_Mm_s_profiles(out_dir, static, ts, active_s0, active_s1)
    plot_overview(out_dir, static, ts, active_s0, active_s1, float(params["WAVE_RAMP_TIME"]))
    write_readme(out_dir, params, geom, args)

    print(f"[OK] saved to: {out_dir}")
    print(f"[OK] input2d: {input_path}")
    print(f"[OK] mesh:    {mesh_path}")
    if args.cpp:
        print(f"[OK] cpp:     {Path(args.cpp).expanduser().resolve()}")
    print(f"[INFO] reference_parameterization = {geom.get('reference_parameterization', 'UNKNOWN')}")
    print(f"[INFO] ref_arc_length = {geom['ref_arc_length']:.8f}")
    print(f"[INFO] ref_backbone_end_x = {geom['ref_backbone_end_x']:.8f}")
    if math.isfinite(float(geom.get("active_end_s_norm", float("nan")))):
        print(f"[INFO] mapped reference-end s_norm = {float(geom['active_end_s_norm']):.8f}")
    print(f"[INFO] active phase coordinate = {params['ACTIVE_PHASE_COORDINATE']}")
    print(f"[INFO] active phase wavelength = {float(params['ACTIVE_PHASE_WAVELENGTH']):.8f}")
    moment_mode = normalize_active_moment_mode(params.get("ACTIVE_MOMENT_MODE", "TRAVELING"))
    print(f"[INFO] ACTIVE_MOMENT_MODE = {moment_mode}")
    k_mode = str(params.get("K_SHAPE_MODE", "HALF-BELL")).upper()
    if moment_mode == "STATIC":
        print("[INFO] active drive = STATIC_MOMENT_M0 inside ACTIVE_S_START < s_norm < ACTIVE_S_END")
    else:
        print("[INFO] active drive = h(s)^2*K_shape(xi)*cos(2*pi*xi/lambda - WAVE_TIME_SIGN*omega*t + ACTIVE_PHASE0)")
    if k_mode == "HALF-BELL":
        print("[INFO] K_shape = HALF-BELL: posterior-rising, 0 at head xi=0, 1 at active-body end")
    else:
        print("[INFO] K_shape = BELL: symmetric bell, zero at head and active-body end")
    print("[INFO] active stress saturation = Mm_max*tanh(Mm_raw/Mm_max)")
    print(f"[INFO] I2 convention = I2_ideal=I2_unit*h^{float(params['ACTIVE_I2_H_POWER']):g}; runtime max(I2_c_FE_scaled, FE_SECTION_I2_FLOOR_RATIO*I2_ideal)")
    print(f"[INFO] active phase-speed direction = {params.get('WAVE_PROPAGATION_S', 'toward increasing phase coordinate (head-to-tail)')}")
    print(f"[INFO] time window = {args.t_start} ~ {args.t_end} s")


if __name__ == "__main__":
    main()
