"""Heuristic terrain score and world-space marker data, independent of Isaac Sim.

This is a visualization heuristic, not a learned LAS output or a success
probability. Missing ray hits have no defensible world-surface position and are
counted as unknown without drawing a misleading terrain marker.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .terrain_affordance import AffordanceMap, CapabilityHypotheses
from .terrain_observation import TerrainObservation

GREEN, YELLOW, RED, GRAY = range(4)
LABELS = ("traversable", "conditional", "non_traversable", "unknown")


@dataclass(frozen=True)
class TraversabilityVizConfig:
    green_threshold: float = 0.7
    yellow_threshold: float = 0.4
    max_distance_m: float = 5.0
    height_offset_m: float = 0.025
    point_radius_m: float = 0.035
    opacity: float = 0.78
    update_hz: float = 10.0
    sample_stride: int = 1

    def __post_init__(self):
        if not (0 <= self.yellow_threshold < self.green_threshold <= 1):
            raise ValueError("thresholds must satisfy 0 <= yellow < green <= 1")
        if self.max_distance_m <= 0 or self.height_offset_m < 0 or self.point_radius_m <= 0:
            raise ValueError("distance/radius must be positive and height offset nonnegative")
        if not isinstance(self.sample_stride, int) or self.sample_stride < 1:
            raise ValueError("sample_stride must be a positive integer")
        if not (0 < self.opacity <= 1) or self.update_hz <= 0:
            raise ValueError("opacity must be in (0, 1] and update_hz positive")


@dataclass
class MarkerFrame:
    positions_w_m: np.ndarray
    categories: np.ndarray
    score: np.ndarray
    grid_categories: np.ndarray
    unknown_count: int
    undrawn_unknown_count: int


def heuristic_traversability(
    obs: TerrainObservation,
    aff: AffordanceMap,
    capability: CapabilityHypotheses = CapabilityHypotheses(),
    config: TraversabilityVizConfig = TraversabilityVizConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Map geometry to [0, 1] using unvalidated capability hypotheses.

    A normalized risk of 1 reaches a capability threshold; risk 2 is twice
    that threshold. Score = clip(1 - 0.5*risk). The default 0.7/0.4 color bands
    begin at 60% and 120% of a geometry threshold, respectively.
    """
    def ratio(values, limit):
        return np.nan_to_num(np.maximum(values, 0.0) / max(limit, 1e-9), nan=0.0, posinf=1e6)

    risk = np.maximum.reduce((
        ratio(aff.slope_deg, capability.max_slope_deg),
        ratio(aff.roughness_m, capability.max_roughness_m),
        ratio(aff.discontinuity_m, capability.max_discontinuity_m),
        np.where(aff.obstacle, ratio(aff.relative_height_m, capability.candidate_step_m), 0.0),
    ))
    score = np.clip(1.0 - 0.5 * risk, 0.0, 1.0)
    score[~obs.valid] = np.nan
    category = np.full(obs.valid.shape, GRAY, dtype=np.int32)
    category[obs.valid & (score >= config.green_threshold)] = GREEN
    category[obs.valid & (score >= config.yellow_threshold) & (score < config.green_threshold)] = YELLOW
    category[obs.valid & (score < config.yellow_threshold)] = RED
    category[aff.gap] = RED  # Only explicitly known unsupported cells are gaps.
    return score, category


def make_marker_frame(
    obs: TerrainObservation,
    aff: AffordanceMap,
    config: TraversabilityVizConfig = TraversabilityVizConfig(),
    capability: CapabilityHypotheses = CapabilityHypotheses(),
) -> MarkerFrame:
    """Use retained raw RayCaster hit XYZ; no reconstructed XY or fictitious Z."""
    if obs.hit_points_w_m is None:
        raise ValueError("world-frame ray hit points are required for viewport markers")
    score, category = heuristic_traversability(obs, aff, capability, config)
    points = obs.hit_points_w_m
    finite = np.isfinite(points).all(axis=2)
    distance = np.linalg.norm(points[..., :2] - np.asarray(obs.base_position_w_m[:2]), axis=2)
    rows, cols = np.indices(obs.valid.shape)
    sampled = (rows % config.sample_stride == 0) & (cols % config.sample_stride == 0)
    shown = finite & sampled & (distance <= config.max_distance_m)
    positions = points[shown].copy()
    positions[:, 2] += config.height_offset_m
    unknown_count = int(np.count_nonzero(category == GRAY))
    return MarkerFrame(
        positions_w_m=positions.astype(np.float32),
        categories=category[shown].astype(np.int32),
        score=score,
        grid_categories=category,
        unknown_count=unknown_count,
        undrawn_unknown_count=int(np.count_nonzero((category == GRAY) & ~finite)),
    )
