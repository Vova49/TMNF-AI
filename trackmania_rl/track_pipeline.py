"""Utilities for working with recorded driving trajectories and track center-line.

This module helps to:
1. Load a recorded trajectory (sequence of 3-D points) that represents the ideal
   racing line or the road centre.
2. Build a *centre-line* representation (piece-wise linear polyline with
   cumulative arc-length parameterisation).
3. Project arbitrary car positions to the centre-line obtaining:
   • s – curvilinear abscissa (progress along the track)  
   • d – lateral deviation from the centre-line (signed, + to the left)  
   • tangent_angle – yaw of the centre-line at the projection point  
   • dist_cp – distance along the centre-line till the next checkpoint (if
     supplied).
4. Provide helper methods used by the RL *TrackmaniaEnv*.

At the moment the implementation is intentionally *very simple* and is based on
piece-wise linear segments in the horizontal (X-Z) plane.  More advanced
representations (e.g. 3-D Catmull–Rom spline) can be plugged in later keeping
exactly the same public API.
"""
from __future__ import annotations

import pathlib
from typing import Tuple, Sequence, Optional

import numpy as np

__all__ = ["CenterLine"]

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class CenterLine:
    """Piece-wise linear centre-line loaded from a text / numpy file.

    The file *path* is expected to contain an ``(N, 3)`` array of XYZ points.
    Supported formats:
    1. ``*.npy`` – NumPy binary with dtype float32 / float64.
    2. ``*.txt`` or ``*.csv`` – ASCII file with whitespace / comma separated
       values, one point per line.

    The coordinate system must match what TMInterface returns (X-forward, Y-up,
    Z-left in game terms).  Only *horizontal* (X, Z) coordinates are used for
    projection computations – elevation (Y) is ignored for now.
    """

    def __init__(self, path: str | pathlib.Path | Sequence[Tuple[float, float, float]]):
        if isinstance(path, (str, pathlib.Path)):
            self._points = self._load_points(pathlib.Path(path))
        else:
            # Already a sequence of points
            self._points = np.asarray(path, dtype=np.float32)

        if self._points.ndim != 2 or self._points.shape[1] != 3:
            raise ValueError("CenterLine expects (N,3) array of XYZ points")

        # Pre-compute cumulative arc-length (s) along the polyline
        diffs = np.diff(self._points[:, [0, 2]], axis=0)  # only X,Z
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self._s = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        self.length: float = float(self._s[-1])

        # Build direction (tangent) angles for each segment in horizontal plane
        # angle w.r.t +X axis, ccw positive when Z increases (right-handed)
        self._seg_angles = np.arctan2(diffs[:, 1], diffs[:, 0])

        # Pre-compute segment unit direction vectors for projection
        with np.errstate(divide="ignore", invalid="ignore"):
            self._seg_dirs = np.divide(diffs, seg_lengths[:, None], where=seg_lengths[:, None] != 0)
            # Replace NaNs (zero length segments) with previous valid direction
            for i in range(1, len(self._seg_dirs)):
                if not np.isfinite(self._seg_dirs[i]).all():
                    self._seg_dirs[i] = self._seg_dirs[i - 1]
            if not np.isfinite(self._seg_dirs[0]).all():
                self._seg_dirs[0] = np.array([1.0, 0.0])  # fallback

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_points(path: pathlib.Path) -> np.ndarray:
        ext = path.suffix.lower()
        if ext == ".npy":
            return np.load(path)

        if ext in {".txt", ".csv"}:
            # читаем текст и нормализуем разделители к пробелам
            with path.open("r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            # если первая строка выглядит как заголовок (есть буквы) — скипаем
            if lines and any(ch.isalpha() for ch in lines[0]):
                lines = lines[1:]

            # заменяем ';' и ',' на пробел — loadtxt съест любое количество пробелов/таба
            norm = "\n".join(lines).replace(";", " ").replace(",", " ")

            from io import StringIO
            arr = np.loadtxt(StringIO(norm), dtype=np.float32)
            return arr

        raise ValueError(f"Unsupported centre-line file format: {path}")

    # ------------------------------------------------------------------
    # Public API used by env.TrackmaniaEnv
    # ------------------------------------------------------------------
    def project(self, pos: Tuple[float, float, float]) -> Tuple[float, float]:
        """Project a *pos* onto the centre-line (horizontal plane).

        Returns (s, d):
            s – curvilinear abscissa (metres from start along centre-line)  
            d – *signed* lateral distance (metres) from the centre-line.  + sign
                corresponds to *left* side with respect to track direction (CCW
                rotation in X-Z plane).
        """
        px, _, pz = pos
        point = np.array([px, pz], dtype=np.float32)

        # Find nearest segment via brute force – can be optimised (KD-tree)
        min_dist = float("inf")
        best_idx = 0
        for i in range(len(self._points) - 1):
            a = self._points[i, [0, 2]]
            b = self._points[i + 1, [0, 2]]
            seg_vec = b - a
            seg_len_sq = float(np.dot(seg_vec, seg_vec))
            if seg_len_sq == 0.0:
                continue
            t = np.clip(np.dot(point - a, seg_vec) / seg_len_sq, 0.0, 1.0)
            proj = a + t * seg_vec
            dist = float(np.linalg.norm(point - proj))
            if dist < min_dist:
                min_dist = dist
                best_idx = i
                best_t = t  # type: ignore
                best_proj = proj  # type: ignore

        # Compute s coordinate at projection point
        s = self._s[best_idx] + best_t * (self._s[best_idx + 1] - self._s[best_idx])

        # Lateral sign via cross product (z axis up in X-Z plane?)
        seg_dir = self._seg_dirs[best_idx]
        rel = point - self._points[best_idx, [0, 2]]
        cross = seg_dir[0] * rel[1] - seg_dir[1] * rel[0]
        d = min_dist * (1.0 if cross >= 0 else -1.0)
        return float(s), float(d)

    def tangent_angle(self, s: float) -> float:
        """Approximate tangent angle (yaw, radians) at curvilinear abscissa *s*."""
        # Binary search to locate segment
        idx = np.searchsorted(self._s, s) - 1
        idx = int(np.clip(idx, 0, len(self._seg_angles) - 1))
        return float(self._seg_angles[idx])

    def dist_to_next_checkpoint(self, s: float, next_cp_s: float) -> float:
        """Distance along the centre-line until *next_cp_s* (handles lap wrap)."""
        if next_cp_s < s:  # crossed finish, wrap around
            return (self.length - s) + next_cp_s
        return next_cp_s - s

    # Convenience helper used by env._state_to_obs
    def project_with_extras(
            self, pos: Tuple[float, float, float], next_cp_s: Optional[float] | None = None
    ) -> Tuple[float, float, float, float]:
        s, d = self.project(pos)
        tang_ang = self.tangent_angle(s)
        dist_cp = self.dist_to_next_checkpoint(s, next_cp_s or self.length)
        return s, d, tang_ang, dist_cp
