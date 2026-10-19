"""Axisymmetric loop Green functions: psi = R A_phi, in Wb/rad per ampere.

Coordinates are metres. Positive current circulates in +phi. A conductor's
total ampere-turns are distributed uniformly over its cross-section.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
import warnings
import numpy as np
from scipy.special import ellipk, ellipe

MU0 = 4e-7 * np.pi
_COIL_CACHE = OrderedDict()
_CACHE_DIR = Path(__file__).resolve().parents[1] / '_work' / 'cache'


def cached_coils(kind, row, R, Z, order, compute):
    """Cache exact geometry/grid results; retain only eight matrices in RAM."""
    digest = hashlib.sha256(b'coil-greens-v1')
    # Invalidate on numerical implementation changes as well as input changes.
    for source in (Path(__file__), Path(__file__).resolve().parents[1]/'mast/simple.py'):
        digest.update(source.read_bytes())
    digest.update(kind.encode())
    digest.update(str(order).encode())
    for value in (R, Z, *(row[k] for k in ('coil_input_column', 'coil_R', 'coil_Z',
                  'coil_width', 'coil_height', 'coil_angle1', 'coil_angle2'))):
        a = np.ascontiguousarray(value)
        digest.update(str((a.dtype.str, a.shape)).encode())
        digest.update(a.tobytes())
    key = digest.hexdigest()
    if key in _COIL_CACHE:
        _COIL_CACHE.move_to_end(key)
        return _COIL_CACHE[key]
    path = _CACHE_DIR / (kind + '-' + key + '.npz')
    matrix = None
    try:
        with np.load(path, allow_pickle=False) as saved:
            candidate = saved['matrix']
            checksum = hashlib.sha256(candidate.tobytes()).hexdigest()
            if (str(saved['key']) == key and str(saved['sha256']) == checksum
                    and candidate.dtype == np.float64
                    and candidate.shape == (len(R)*len(Z), len(set(row['coil_input_column']))
                                            if kind == 'mast' else len(row['coil_input_column']))
                    and np.isfinite(candidate).all()):
                matrix = candidate
    except (OSError, ValueError, KeyError, EOFError):
        pass
    if matrix is None:
        matrix = compute()
        if not np.isfinite(matrix).all():
            raise ValueError('Nonfinite coil Green matrix')
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # Exclusive creation; interrupted or invalid files are never reused.
            with path.open('xb') as stream:
                np.savez(stream, key=np.asarray(key), matrix=matrix,
                         sha256=np.asarray(hashlib.sha256(matrix.tobytes()).hexdigest()))
        except FileExistsError:
            pass
        except OSError as exc:
            warnings.warn(f'Coil cache not saved: {exc}', RuntimeWarning)
    matrix.setflags(write=False)
    _COIL_CACHE[key] = matrix
    while len(_COIL_CACHE) > 8:
        _COIL_CACHE.popitem(last=False)
    return matrix


def loop_flux(R, Z, source_R, source_Z):
    """Broadcast psi/I = mu0 sqrt(R a)/(2 pi k) [(2-k²)K(k²)-2E(k²)].

    A zero-radius observation is exactly zero. At a source filament the
    logarithmic singularity is represented by +inf, never silently softened.
    """
    R, Z, a, b = np.broadcast_arrays(
        np.asarray(R, float), np.asarray(Z, float),
        np.asarray(source_R, float), np.asarray(source_Z, float))
    if (R < 0).any() or (a <= 0).any():
        raise ValueError("Observation R must be nonnegative; source R positive")
    d2 = (R + a)**2 + (Z - b)**2
    m = np.clip(4 * R * a / d2, 0, 1)
    # Rewrite sqrt(Ra)/k as sqrt(d2)/2. For small m use a series
    # to avoid cancellation of two almost equal elliptic integrals.
    small = m < 1e-3
    bracket = np.empty_like(m)
    x = m[small]
    bracket[small] = np.pi / 16 * x*x * (
        1 + .75*x + 75/128*x*x + 245/512*x**3 + 6615/16384*x**4)
    x = m[~small]
    bracket[~small] = (2-x)*ellipk(x) - 2*ellipe(x)
    return MU0 / (4*np.pi) * np.sqrt(d2) * bracket


def rectangle_nodes(R, Z, width, height, angle1=0., angle2=0., order=6):
    """Uniform-current cross-section quadrature using Gauss-Legendre nodes.

    EFIT AF/AF2 interpretation used here: AF shears R along Z and AF2
    shears Z along R, with cot(angle); zero or 90 degrees means no shear.
    README names the angles but does not give this coordinate transform.
    This interpretation must be checked against an EFIT implementation before
    treating nonzero-skew numerical results as a verified physical upper bound.
    """
    if width <= 0 or height <= 0 or order < 1:
        raise ValueError("Positive conductor dimensions and quadrature order required")
    x, w = np.polynomial.legendre.leggauss(order)
    u, v = np.meshgrid(x * width/2, x * height/2)
    weights = np.outer(w, w).ravel()/4

    def shear(angle):
        if np.isclose(angle % 180, 0) or np.isclose(angle % 180, 90):
            return 0.
        return 1 / np.tan(np.deg2rad(angle))

    a = R + u + shear(angle1)*v
    b = Z + v + shear(angle2)*u
    if (a <= 0).any():
        raise ValueError("Conductor extends to nonpositive R")
    return a.ravel(), b.ravel(), weights


def rectangle_flux(R, Z, source_R, source_Z, width, height,
                   angle1=0., angle2=0., order=6):
    """psi per total A-turn of a rectangle; weights sum to one, not area."""
    a, b, w = rectangle_nodes(source_R, source_Z, width, height,
                              angle1, angle2, order)
    out = np.zeros(np.broadcast_shapes(np.shape(R), np.shape(Z)))
    # Do not allocate observation-grid x conductor-elements x time.
    for ar, bz, weight in zip(a, b, w):
        out += weight * loop_flux(R, Z, ar, bz)
    return out


def coil_matrix(row, R, Z, order=6):
    """Pixel x geometry-row matrix, independent of coil-current column order."""
    rr, zz = np.meshgrid(R, Z)
    columns = list(row["coil_input_column"])
    if any(c == "magnetics_bcoil" for c in columns):
        raise ValueError("Toroidal field coil must not enter poloidal-flux sum")
    return cached_coils('diii_d', row, R, Z, order, lambda: np.column_stack([
        rectangle_flux(rr, zz, row["coil_R"][j], row["coil_Z"][j],
                       row["coil_width"][j], row["coil_height"][j],
                       row["coil_angle1"][j], row["coil_angle2"][j], order).ravel()
        for j in range(len(columns))]))


def plasma_grid(nR=12, nZ=16, bounds=(1.15, 2.15, -.8, .8)):
    """Cell centres in an explicit (Rmin,Rmax,Zmin,Zmax) box, in metres."""
    r0, r1, z0, z1 = bounds
    if nR < 1 or nZ < 1 or not (0 < r0 < r1 and z0 < z1):
        raise ValueError("Invalid plasma box or source count")
    r = r0 + (np.arange(nR)+.5)*(r1-r0)/nR
    z = z0 + (np.arange(nZ)+.5)*(z1-z0)/nZ
    a, b = np.meshgrid(r, z)
    return a.ravel(), b.ravel()


def plasma_matrix(R, Z, nR=12, nZ=16, bounds=(1.15, 2.15, -.8, .8)):
    rr, zz = np.meshgrid(R, Z)
    a, b = plasma_grid(nR, nZ, bounds)
    g = loop_flux(rr.ravel()[:, None], zz.ravel()[:, None], a, b)
    if not np.isfinite(g).all():
        raise ValueError("Plasma source coincides with an EFIT grid point")
    return g
