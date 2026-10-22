"""Label-free q and pressure physics. No data reads, writes, or command entrypoint.

psi is Wb/rad, so Bp = |grad psi|/R (no extra 2*pi).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# The published scorer uses sibling imports; do not modify its source files.
_SCORER = str(Path(__file__).resolve().parents[1] / 'starter_ref/fusion_scoring')
if _SCORER not in sys.path:
    sys.path.insert(0, _SCORER)
from derive import _phi, magnetic_axis, _shape_from_contour, _bilinear, _points_in_polygon
from lcfs import extract_lcfs_with_sign
from contour import find_contours, grid_to_physical, is_closed, encloses_point

MU0 = 4e-7 * np.pi
EV_J = 1.602e-19
# Published engineering turn counts (machine hardware, not equilibrium data).
# MAST: 24 TF turns -- A. C. Darke et al., "MAST: The Mega Amp Spherical Tokamak"
# ("MAST uses same LVPS supply, in 24 turns giving 0.6T"); consistent with the
# 2.2 MA design TF rod current and 0.52 T at R = 0.85 m in A. Sykes, "First Results
# From MAST", IAEA FEC 2000, OV4/1, Table 1. DIII-D: 144 TF turns (24 bundles of
# six) -- E. E. Reis et al., GA-A-13668 (1975). starter_ref/README.md:721,821 gives
# the TF currents in kA but no turn counts. Never fitted to either machine's labels.
TF_TURNS = {'MAST': 24, 'DIII-D': 144}

# Physical ranges (general tokamak physics, not MAST labels): q95 < 2 is the
# external-kink disruption limit; q95 > 20 or betaN > 7 (above ideal-wall limits even
# for spherical tokamaks) marks a numerical breakdown of the estimate on a crude flux
# map. Values outside are treated as failed frames and filled like other failures.
PHYSICAL_RANGE = {'q95': (2.0, 20.0), 'betaN': (0.0, 7.0)}


def in_range(field, values):
    """Finite and strictly inside PHYSICAL_RANGE[field]."""
    lo, hi = PHYSICAL_RANGE[field]
    values = np.asarray(values, float)
    with np.errstate(invalid='ignore'):
        return np.isfinite(values) & (values > lo) & (values < hi)


def fill_out_of_range(field, values, fallback):
    """Replace failed or out-of-range frames by the in-range median of the same shot,
    or by `fallback` when the shot has none (the policy of mast/qb_generate.py)."""
    values = np.asarray(values, float)
    ok = in_range(field, values)
    value = float(np.median(values[ok])) if ok.any() else float(fallback)
    return np.where(ok, values, value), int((~ok).sum())


def vacuum_F(i_tf_kA, n_turns):
    """Vacuum R*Bt in T*m, from TF conductor current in kA and explicit turns."""
    return MU0 * n_turns * np.abs(np.asarray(i_tf_kA, float)) * 1e3 / (2*np.pi)


def _grid(psi, R, Z):
    p, r, z = np.asarray(psi, float), np.asarray(R, float), np.asarray(Z, float)
    if (r.ndim != 1 or z.ndim != 1 or min(len(r), len(z)) < 3
            or p.shape != (len(z), len(r)) or not np.isfinite(p).all()
            or not np.isfinite(r).all() or not np.isfinite(z).all()
            or not np.all(np.diff(r) > 0) or not np.all(np.diff(z) > 0)
            or np.min(r) <= 0 or np.ptp(p) == 0):
        raise ValueError('Invalid physical flux grid')
    return p, r, z


def flux_geometry(psi, R, Z, machine='MAST', mask_coarse=None, mask_f=None,
                  axis_sign=None, n_points=512):
    """Scorer LCFS/O-point geometry; scalar NaNs and lcfs=None on failure.

    The axis flux is the peak cell used by lcfs._extract_one_sign. LCFS flux
    is the median bilinear flux on the scorer's resampled polygon. This small
    discretisation approximation avoids changing the published extractor.
    No MAST envelope (derived from labels) is used by callers.
    """
    out = dict(phi_axis=np.nan, phi_lcfs=np.nan, lcfs=None, R_geo=np.nan,
               a=np.nan, volume=np.nan, R_axis=np.nan, Z_axis=np.nan,
               iz=None, ir=None, axis_sign=np.nan)
    try:
        p, r, z = _grid(psi, R, Z)
        # Same defaults, masks, point count and extractor as derive_frame.
        C, sign = extract_lcfs_with_sign(p, r, z, machine, mask_coarse,
                                        mask_f, n_points=n_points, axis_sign=axis_sign)
        if C is None or len(C) < 8:
            return out
        ra, za, iz, ir = magnetic_axis(p, r, z, machine, mask_coarse, sign)
        phi = _phi(p, machine, sign)
        pa = float(phi[iz, ir])
        pb = float(np.median(_bilinear(phi, r, z, C[:, 0], C[:, 1])))
        shape = _shape_from_contour(C)
        a = float(np.ptp(C[:, 0])/2)
        rg = float((C[:, 0].max()+C[:, 0].min())/2)
        if not (np.isfinite([pa, pb, a, rg, shape['volume']]).all()
                and pa > pb and a > 0 and rg > a and shape['volume'] > 0):
            return out
        out.update(phi_axis=pa, phi_lcfs=pb, lcfs=C, R_geo=rg, a=a,
                   volume=shape['volume'], R_axis=ra, Z_axis=za, iz=iz, ir=ir,
                   axis_sign=sign)
    except Exception:
        pass  # A failed frame is represented by NaNs, never a guessed geometry.
    return out


def _surface(phi, R, Z, geom, psin):
    level = geom['phi_axis'] + psin*(geom['phi_lcfs']-geom['phi_axis'])
    for c in find_contours(phi, level):
        if (len(c) >= 4 and is_closed(c)
                and encloses_point(c, geom['iz'], geom['ir'])):
            return grid_to_physical(c, R, Z)
    return None


def q_at(psi, R, Z, F, psin=0.95, machine='MAST', mask_coarse=None,
         mask_f=None, geometry=None, axis_sign=None):
    """|F|/(2*pi) integral dl/(R*|grad psi|) on a closed axis-enclosing surface."""
    try:
        p, r, z = _grid(psi, R, Z)
        if not (0 < psin < 1 and np.isfinite(F) and abs(F) > 0):
            return np.nan
        g = geometry if geometry is not None else flux_geometry(
            p, r, z, machine, mask_coarse, mask_f, axis_sign)
        if g['lcfs'] is None:
            return np.nan
        C = _surface(_phi(p, machine, g['axis_sign']), r, z, g, psin)
        if C is None:
            return np.nan
        dz, dr = np.gradient(p, z, r)
        grad = np.hypot(dr, dz)
        # Marching-squares returns the closing point too. Integrate at segment midpoints.
        mid = (C[:-1]+C[1:])/2
        dl = np.linalg.norm(np.diff(C, axis=0), axis=1)
        den = mid[:, 0]*_bilinear(grad, r, z, mid[:, 0], mid[:, 1])
        if not (np.isfinite(den).all() and np.all(den > 0)):
            return np.nan
        value = float(abs(F)/(2*np.pi)*np.sum(dl/den))
        return value if np.isfinite(value) and value > 0 else np.nan
    except Exception:
        return np.nan


def volume_average_pressure(psi, R, Z, sample_R, sample_Z, Te_eV, ne_m3,
                            machine='MAST', mask_coarse=None, mask_f=None,
                            geometry=None, n_bins=20):
    """Electron pressure [Pa], averaged in flux bins then weighted by shell volume.

    Only finite, positive Te/ne inside the LCFS and grid are used. Empty bins
    interpolate between occupied-bin centres; unmeasured ends take the nearest
    bin value. At least two occupied bins are required. This explicit profile
    extrapolation is an assumption, not a measurement of the missing centre/edge.
    """
    try:
        p, r, z = _grid(psi, R, Z)
        g = geometry if geometry is not None else flux_geometry(
            p, r, z, machine, mask_coarse, mask_f)
        if g['lcfs'] is None or n_bins < 2:
            return np.nan
        sr, sz, te, ne = np.broadcast_arrays(*[np.asarray(x, float) for x in
                                               (sample_R, sample_Z, Te_eV, ne_m3)])
        sr, sz, te, ne = [x.ravel() for x in (sr, sz, te, ne)]
        ok = (np.isfinite(sr+sz+te+ne) & (te > 0) & (ne > 0)
              & (sr >= r[0]) & (sr <= r[-1]) & (sz >= z[0]) & (sz <= z[-1]))
        ok &= _points_in_polygon(sr, sz, g['lcfs'])
        phi = _phi(p, machine, g['axis_sign'])
        pn = (_bilinear(phi, r, z, sr[ok], sz[ok])-g['phi_axis']) / (
            g['phi_lcfs']-g['phi_axis'])
        pe = te[ok]*ne[ok]*EV_J
        valid = (pn >= 0) & (pn <= 1) & np.isfinite(pe)
        pn, pe = pn[valid], pe[valid]
        bins = np.minimum((pn*n_bins).astype(int), n_bins-1)
        counts = np.bincount(bins, minlength=n_bins)
        sums = np.bincount(bins, weights=pe, minlength=n_bins)
        occupied = counts > 0
        if occupied.sum() < 2:
            return np.nan
        centres = (np.arange(n_bins)+0.5)/n_bins
        profile = np.interp(centres, centres[occupied], sums[occupied]/counts[occupied])
        volumes = [0.0]
        for pn_edge in np.arange(1, n_bins)/n_bins:
            C = _surface(phi, r, z, g, pn_edge)
            if C is None:
                return np.nan
            volumes.append(_shape_from_contour(C)['volume'])
        volumes.append(g['volume'])
        dv = np.diff(volumes)
        if not np.isfinite(dv).all() or np.any(dv <= 0):
            return np.nan
        result = float(np.dot(profile, dv)/g['volume'])
        return result if np.isfinite(result) and result > 0 else np.nan
    except Exception:
        return np.nan


def beta_n(p_total, a, B0, ip_A):
    """Normalised beta; pressure in Pa, a in m, B0 in T, signed Ip in A.

    The denominator follows the requested formula literally. For physical betaN,
    callers explicitly supply the current strength abs(Ip), in amperes.
    """
    try:
        if (not np.isfinite([p_total, a, B0, ip_A]).all() or p_total < 0
                or min(a, B0, abs(ip_A)) <= 0):
            return np.nan
        value = 100*(2*MU0*p_total/B0**2)*a*B0/(ip_A/1e6)
        return float(value) if np.isfinite(value) else np.nan
    except Exception:
        return np.nan
