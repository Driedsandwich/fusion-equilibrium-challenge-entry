"""Causal diagnostic summaries on the official, corrected input time axes (ms)."""
from __future__ import annotations

import numpy as np
from features import SIGNALS, INPUT_COLUMNS, fix_d3d_ip_times

WINDOWS_MS = (10., 50.)


def feature_names():
    names = list(SIGNALS)
    names += [f"{s}_{int(w)}ms_{stat}" for s in SIGNALS for w in WINDOWS_MS
              for stat in ("mean", "slope_per_second")]
    names += [f"thomson_{system}_{measure}_{stat}"
              for system in ("core", "edge") for measure in ("Te", "ne")
              for stat in (("peak", "mean", "peak_channel01", "width_channel01", "std")
                           if system == "core" else ("mean", "peak", "std"))]
    return names


def latest(times, values, target):
    times, values = np.asarray(times, float), np.asarray(values, float)
    if len(times) != len(values) or np.any(np.diff(times) <= 0) or not np.isfinite(times).all():
        raise ValueError("Invalid diagnostic time axis")
    idx = np.searchsorted(times, target, side="right") - 1
    out = np.zeros((len(target),) + values.shape[1:])
    valid = idx >= 0
    out[valid] = values[idx[valid]]
    return out


def history(times, values, target):
    """Sample means and least-squares slopes in [t-w,t]; no extrapolation."""
    times, values = np.asarray(times, float), np.asarray(values, float)
    valid = np.isfinite(times) & np.isfinite(values)
    times, values = times[valid], values[valid]
    if np.any(np.diff(times) <= 0):
        raise ValueError("Nonmonotone current times")
    now = latest(times, values, target)
    cols = []
    # Centred seconds reduce cancellation in the prefix sums for short windows.
    ts = (times - times[0]) / 1000. if len(times) else times
    sums = [np.r_[0., np.cumsum(v)] for v in (values, ts, ts*ts, ts*values)]
    right = np.searchsorted(times, target, side="right")
    for width in WINDOWS_MS:
        left = np.searchsorted(times, target-width, side="left")
        n = right-left
        sy, st, stt, sty = [v[right]-v[left] for v in sums]
        mean = np.divide(sy, n, out=np.zeros(len(target)), where=n > 0)
        denom = stt - np.divide(st*st, n, out=np.zeros(len(target)), where=n > 0)
        numer = sty - np.divide(st*sy, n, out=np.zeros(len(target)), where=n > 0)
        slope = np.divide(numer, denom, out=np.zeros(len(target)), where=(n > 1) & (denom > 1e-14))
        cols.extend((mean, slope))
    return now, cols


def profile_summary(profiles, core):
    profiles = np.asarray(profiles, float)
    if profiles.ndim != 2 or profiles.shape[1] == 0:
        raise ValueError("Expected a nonempty Thomson time/channel matrix")
    finite = np.isfinite(profiles)
    count = finite.sum(axis=1)
    clean = np.where(finite, profiles, 0.)
    mean = clean.sum(axis=1) / np.maximum(count, 1)
    std = np.sqrt((np.where(finite, (clean-mean[:, None])**2, 0.)).sum(axis=1) / np.maximum(count, 1))
    peak = np.max(np.where(finite, profiles, -np.inf), axis=1)
    peak[count == 0] = 0.
    if not core:
        return np.column_stack((mean, peak, std))
    coord = np.linspace(0., 1., profiles.shape[1])
    pos = coord[np.argmax(np.where(finite, profiles, -np.inf), axis=1)]
    weight = np.maximum(clean, 0.)
    mass = weight.sum(axis=1)
    centre = (weight*coord).sum(axis=1) / np.maximum(mass, 1e-30)
    width = np.sqrt((weight*(coord-centre[:, None])**2).sum(axis=1) / np.maximum(mass, 1e-30))
    pos[count == 0] = 0.
    return np.column_stack((peak, mean, pos, width, std))


def causal_ip_history(row, target):
    """Apply the official fixer only to samples past under BOTH candidate axes.

    The shipped timestamps can be wrong. Before deciding between the shipped
    axis and its official re-origining, looking at either candidate's whole
    trace could leak future measurements under the other candidate. Using the
    intersection is conservative: Ip may be stale or absent in short windows.
    """
    shipped = np.asarray(row["magnetics_plasma_current_times"], float)
    values = np.asarray(row["magnetics_plasma_current"], float)
    origin = float(row["magnetics_time"][0])
    if len(shipped) == 0 or len(shipped) != len(values) or not np.isfinite(shipped).all() or np.any(np.diff(shipped) <= 0):
        raise ValueError("Invalid Ip acquisition axis")
    shifted = origin + shipped-shipped[0]
    safe_axis = np.maximum(shipped, shifted)
    out = np.zeros((len(target), 5))
    for i, t in enumerate(target):
        n = int(np.searchsorted(safe_axis, t, side="right"))
        if not n:
            continue
        prefix = {"magnetics_plasma_current_times": shipped[:n],
                  "magnetics_plasma_current": values[:n], "magnetics_time": [origin]}
        fixed = fix_d3d_ip_times(prefix)
        now, windows = history(fixed, values[:n], np.array([t]))
        out[i] = [now[0], *(v[0] for v in windows)]
    return out[:, 0], [out[:, j] for j in range(1, 5)]


def build_features(row, target=None):
    # EFIT labels, geometry, and the scoring envelope are never read here.
    target = np.asarray(row["efit_times"] if target is None else target, float)
    instantaneous, windows = [], []
    for signal in SIGNALS:
        if signal == "plasma_current":
            now, cols = causal_ip_history(row, target)
        else:
            now, cols = history(row["magnetics_time"], row[f"magnetics_{signal}"], target)
        instantaneous.append(now)
        windows.extend(cols)
    blocks = [np.column_stack(instantaneous + windows)]
    for system in ("core", "edge"):
        for measure in ("Te", "ne"):
            key = f"thomson_{system}_{measure}"
            if row.get(key) is None:
                blocks.append(np.zeros((len(target), 5 if system == "core" else 3)))
            else:
                summary = profile_summary(row[key], system == "core")
                blocks.append(latest(row[f"thomson_{system}_times"], summary, target))
    result = np.column_stack(blocks).astype(np.float32)
    if result.shape[1] != len(feature_names()) or not np.isfinite(result).all():
        raise ValueError("Invalid feature table")
    return result
