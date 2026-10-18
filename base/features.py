"""Input-only DIII-D features, adapted from starter_ref/experiments.py.

Only diagnostic inputs are interpolated. EFIT time is the requested output axis;
no EFIT target or scoring mask is inspected here. Native input units are retained.
"""
from __future__ import annotations

import sys
from pathlib import Path
import warnings

import numpy as np
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter_ref"))
from data_fixes import fix_d3d_ip_times

SIGNALS = ["ECOILA", "bcoil", *[f"F{i}{ab}" for i in range(1, 10) for ab in "AB"],
           "plasma_current"]
INPUT_COLUMNS = ["efit_times", "magnetics_time", "magnetics_plasma_current_times",
                 *[f"magnetics_{s}" for s in SIGNALS],
                 *[f"thomson_{s}_{v}" for s in ("core", "edge")
                   for v in ("times", "Te", "ne")]]


def interpolate(times, values, target, outside="extrapolate"):
    times, values = np.asarray(times, float), np.asarray(values, float)
    good = np.isfinite(times) & np.isfinite(values)
    if good.sum() < 2:
        return np.zeros(len(target))
    times, values = times[good], values[good]
    if np.any(np.diff(times) <= 0):
        raise ValueError("Input times must be strictly increasing")
    return interp1d(times, values, bounds_error=False, fill_value=outside)(target)


def feature_names(include_thomson=False):
    names = list(SIGNALS)
    if include_thomson:
        names += [f"thomson_{s}_{v}_{stat}" for s in ("core", "edge")
                  for v in ("Te", "ne") for stat in ("mean", "std", "max")]
    return names


def build_features(row, include_thomson=False):
    """Return a fixed-width (EFIT frames, features) table; no target interpolation."""
    target = np.asarray(row["efit_times"], float)
    ip_times = fix_d3d_ip_times(row)
    cols = [interpolate(ip_times if s == "plasma_current" else row["magnetics_time"],
                        row[f"magnetics_{s}"], target) for s in SIGNALS]
    if include_thomson:
        for system in ("core", "edge"):
            for measure in ("Te", "ne"):
                key = f"thomson_{system}_{measure}"
                if key not in row or row[key] is None:
                    cols.extend([np.zeros(len(target)) for _ in range(3)])
                    continue
                profiles = np.asarray(row[key], float)
                if profiles.ndim != 2:
                    raise ValueError(f"Expected time/channel matrix: {key}")
                profiles = np.where(np.isfinite(profiles), profiles, np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    summaries = (np.nanmean(profiles, axis=1),
                                 np.nanstd(profiles, axis=1), np.nanmax(profiles, axis=1))
                cols.extend(interpolate(row[f"thomson_{system}_times"], v, target, 0.)
                            for v in summaries)
    result = np.column_stack(cols).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite features; refusing to drop held-out frames")
    return result
