from __future__ import annotations

import numpy as np
import torch
from scipy.fft import dct
from scipy.interpolate import CubicSpline, interp1d, make_interp_spline, make_lsq_spline


def _as_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _unique_sorted(s, z):
    s = _as_numpy(s).reshape(-1).astype(np.float64)
    z = _as_numpy(z).astype(np.float64)
    order = np.argsort(s)
    s = s[order]
    z = z[order]
    keep = np.ones(len(s), dtype=bool)
    keep[1:] = np.diff(s) > 1e-8
    return s[keep], z[keep]


class InterpolationBaseline:
    def __init__(self, kind="linear"):
        self.kind = kind
        self.fn = None

    def fit(self, s, z):
        s_np, z_np = _unique_sorted(s, z)
        if len(s_np) < 2:
            self.fn = lambda query: np.repeat(z_np[:1], len(query), axis=0)
        elif self.kind == "cubic" and len(s_np) >= 4:
            self.fn = CubicSpline(s_np, z_np, axis=0, extrapolate=True)
        else:
            self.fn = interp1d(
                s_np,
                z_np,
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
                assume_sorted=True,
            )
        return self

    def predict(self, s, device=None, dtype=torch.float32):
        q = _as_numpy(s).reshape(-1).astype(np.float64)
        out = np.asarray(self.fn(q), dtype=np.float32)
        return torch.as_tensor(out, dtype=dtype, device=device)


class BSplineBaseline:
    def __init__(self, degree=3, control_points=16):
        self.degree = int(degree)
        self.control_points = int(control_points)
        self.fn = None

    def fit(self, s, z):
        s_np, z_np = _unique_sorted(s, z)
        degree = min(self.degree, max(len(s_np) - 1, 1))
        if len(s_np) <= degree + 1:
            self.fn = make_interp_spline(s_np, z_np, k=degree, axis=0)
            return self

        n_coeff = min(max(degree + 1, self.control_points), len(s_np))
        interior_count = max(n_coeff - degree - 1, 0)
        if interior_count <= 0:
            knots = np.r_[np.repeat(s_np[0], degree + 1), np.repeat(s_np[-1], degree + 1)]
        else:
            quantiles = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
            interior = np.quantile(s_np, quantiles)
            knots = np.r_[np.repeat(s_np[0], degree + 1), interior, np.repeat(s_np[-1], degree + 1)]
        try:
            self.fn = make_lsq_spline(s_np, z_np, knots, k=degree, axis=0)
        except Exception:
            self.fn = make_interp_spline(s_np, z_np, k=degree, axis=0)
        return self

    def predict(self, s, device=None, dtype=torch.float32):
        q = _as_numpy(s).reshape(-1).astype(np.float64)
        out = np.asarray(self.fn(q), dtype=np.float32)
        return torch.as_tensor(out, dtype=dtype, device=device)


class DCTBaseline:
    def __init__(self, components=32, ridge=1e-4):
        self.components = int(components)
        self.ridge = float(ridge)
        self.coeff = None

    @staticmethod
    def basis(s, components):
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        u = (s - s.min()) / max(float(s.max() - s.min()), 1e-8)
        cols = [np.ones_like(u)]
        for k in range(1, components):
            cols.append(np.cos(np.pi * k * u))
        return np.stack(cols, axis=1)

    def fit(self, s, z):
        s_np, z_np = _unique_sorted(s, z)
        comps = min(max(1, self.components), len(s_np))
        design = self.basis(s_np, comps)
        gram = design.T @ design
        if self.ridge > 0:
            penalty = np.eye(gram.shape[0], dtype=gram.dtype) * self.ridge
            penalty[0, 0] = 0.0
            gram = gram + penalty
        self.coeff = np.linalg.solve(gram, design.T @ z_np)
        self.components = comps
        self.s_min = float(s_np.min())
        self.s_max = float(s_np.max())
        return self

    def predict(self, s, device=None, dtype=torch.float32):
        q = _as_numpy(s).reshape(-1).astype(np.float64)
        u = (q - self.s_min) / max(self.s_max - self.s_min, 1e-8)
        cols = [np.ones_like(u)]
        for k in range(1, self.components):
            cols.append(np.cos(np.pi * k * u))
        design = np.stack(cols, axis=1)
        out = design @ self.coeff
        return torch.as_tensor(out.astype(np.float32), dtype=dtype, device=device)


def build_baseline(name, **kwargs):
    name = str(name).lower()
    if name in {"linear", "linear_interp", "linear_interpolation"}:
        return InterpolationBaseline(kind="linear")
    if name in {"cubic", "cubic_spline"}:
        return InterpolationBaseline(kind="cubic")
    if name in {"bspline", "b_spline", "b-spline"}:
        return BSplineBaseline(
            degree=kwargs.get("degree", 3),
            control_points=kwargs.get("control_points", 16),
        )
    if name == "dct":
        return DCTBaseline(components=kwargs.get("components", 32), ridge=kwargs.get("ridge", 1e-4))
    raise ValueError(f"Unsupported baseline model={name!r}")
