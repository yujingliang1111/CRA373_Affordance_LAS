"""Isaac-independent terrain observations. Grid rows are forward (+x), columns left (+y).

Heights are world-up elevations in metres. The grid rotates with base yaw, but does
not rotate with base roll/pitch; this keeps slope relative to gravity meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import numpy as np


@dataclass
class TerrainObservation:
    height_m: np.ndarray
    valid: np.ndarray
    resolution_m: float
    base_position_w_m: tuple[float, float, float] = (0.0, 0.0, 0.32)
    base_yaw_rad: float = 0.0
    base_velocity_b_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rgb: np.ndarray | None = None
    depth_m: np.ndarray | None = None
    known_unsupported: np.ndarray | None = None
    timestamp_s: float = field(default_factory=time.time)
    source: str = "offline_height_grid"
    metadata: dict = field(default_factory=dict)
    hit_points_w_m: np.ndarray | None = None

    def __post_init__(self):
        self.height_m = np.asarray(self.height_m, dtype=float)
        self.valid = np.asarray(self.valid, dtype=bool)
        if self.height_m.ndim != 2 or self.height_m.shape != self.valid.shape:
            raise ValueError("height_m and valid must be matching 2D arrays")
        if self.resolution_m <= 0 or not np.isfinite(self.resolution_m):
            raise ValueError("resolution_m must be positive and finite")
        self.valid &= np.isfinite(self.height_m)
        if self.hit_points_w_m is not None:
            self.hit_points_w_m = np.asarray(self.hit_points_w_m, dtype=float)
            if self.hit_points_w_m.shape != (*self.height_m.shape, 3):
                raise ValueError("hit_points_w_m must have shape (rows, cols, 3)")
            self.valid &= np.isfinite(self.hit_points_w_m).all(axis=2)
        if self.known_unsupported is None:
            self.known_unsupported = np.zeros_like(self.valid)
        else:
            self.known_unsupported = np.asarray(self.known_unsupported, dtype=bool)
            if self.known_unsupported.shape != self.valid.shape:
                raise ValueError("known_unsupported shape differs from height_m")
            self.known_unsupported &= ~self.valid

    @property
    def xy_grid_m(self) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = self.height_m.shape
        x = (np.arange(rows) - (rows - 1) / 2) * self.resolution_m
        y = (np.arange(cols) - (cols - 1) / 2) * self.resolution_m
        return np.meshgrid(x, y, indexing="ij")



def yaw_from_xyzw(quat_xyzw: np.ndarray) -> float:
    """Extract world yaw from Isaac Lab 3 / Warp quaternion order (x, y, z, w)."""
    x, y, z, w = map(float, quat_xyzw)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

def from_isaac_ray_hits(
    ray_hits_w_m: np.ndarray,
    base_position_w_m: tuple[float, float, float],
    base_yaw_rad: float,
    *,
    shape: tuple[int, int] = (17, 11),
    resolution_m: float = 0.1,
    base_velocity_b_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
    timestamp_s: float | None = None,
) -> TerrainObservation:
    """Bin raw world-frame RayCaster hits; never use clipped PPO height observations.

    The source scanner is GridPatternCfg(size=[1.6, 1.0], resolution=0.1).
    Multiple hits per cell use their median. Missing hits remain unknown, not gaps.
    """
    hits = np.asarray(ray_hits_w_m, dtype=float).reshape(-1, 3)
    base = np.asarray(base_position_w_m, dtype=float)
    if base.shape != (3,):
        raise ValueError("base_position_w_m must have three elements")
    c, s = np.cos(base_yaw_rad), np.sin(base_yaw_rad)
    delta = hits[:, :2] - base[:2]
    x = c * delta[:, 0] + s * delta[:, 1]
    y = -s * delta[:, 0] + c * delta[:, 1]
    finite = np.isfinite(hits).all(axis=1)
    ri = np.zeros(len(hits), dtype=int)
    ci = np.zeros(len(hits), dtype=int)
    ri[finite] = np.rint(x[finite] / resolution_m + (shape[0] - 1) / 2).astype(int)
    ci[finite] = np.rint(y[finite] / resolution_m + (shape[1] - 1) / 2).astype(int)
    inside = finite & (ri >= 0) & (ri < shape[0]) & (ci >= 0) & (ci < shape[1])
    grid = np.full(shape, np.nan)
    valid = np.zeros(shape, bool)
    points = np.full((*shape, 3), np.nan)
    buckets: dict[tuple[int, int], list[np.ndarray]] = {}
    for r, col, hit in zip(ri[inside], ci[inside], hits[inside]):
        buckets.setdefault((r, col), []).append(hit)
    for (r, col), values in buckets.items():
        candidates = np.asarray(values)
        median_z = float(np.median(candidates[:, 2]))
        # Retain an actual measured world point even when several rays bin together.
        representative = candidates[np.argmin(np.abs(candidates[:, 2] - median_z))]
        grid[r, col] = median_z
        points[r, col] = representative
        valid[r, col] = True
    return TerrainObservation(
        grid, valid, resolution_m, tuple(base), base_yaw_rad,
        base_velocity_b_mps, timestamp_s=time.time() if timestamp_s is None else timestamp_s,
        source="isaac_ray_caster", metadata={"frame": "base_yaw_aligned_world_up", "input_hits": len(hits)},
        hit_points_w_m=points,
    )
