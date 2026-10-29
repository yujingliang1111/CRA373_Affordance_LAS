"""Geometry-based affordance maps. Demo thresholds are hypotheses, not robot limits."""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
from .terrain_observation import TerrainObservation

@dataclass(frozen=True)
class CapabilityHypotheses:
    ordinary_step_m: float = 0.06
    candidate_step_m: float = 0.18
    max_slope_deg: float = 15.0
    max_roughness_m: float = 0.035
    max_discontinuity_m: float = 0.07
    min_landing_length_m: float = 0.20
    foot_clearance_cells: int = 1
    climb_controller_validated: bool = False
    jump_controller_validated: bool = False

@dataclass
class AffordanceMap:
    slope_deg: np.ndarray
    roughness_m: np.ndarray
    discontinuity_m: np.ndarray
    relative_height_m: np.ndarray
    traversable: np.ndarray
    non_traversable: np.ndarray
    foothold: np.ndarray
    high_risk: np.ndarray
    obstacle: np.ndarray
    transition: np.ndarray
    gap: np.ndarray
    potentially_crossable: np.ndarray
    unknown: np.ndarray
    baseline_height_m: float

def _local_stats(h: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = []
    padded = np.pad(np.where(valid, h, np.nan), 1, constant_values=np.nan)
    for dr in range(3):
        for dc in range(3):
            samples.append(padded[dr:dr+h.shape[0], dc:dc+h.shape[1]])
    stack = np.stack(samples)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # entirely missing support windows
        return np.nanstd(stack, axis=0), np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)

def estimate_affordance(obs: TerrainObservation, cfg: CapabilityHypotheses = CapabilityHypotheses()) -> AffordanceMap:
    h, valid = obs.height_m, obs.valid
    if not valid.any():
        raise ValueError("observation has no valid height samples")
    rear_n = max(1, h.shape[0] // 3)
    rear = h[:rear_n][valid[:rear_n]]
    baseline = float(np.median(rear if rear.size else h[valid]))
    relative = np.where(valid, h - baseline, np.nan)
    gx, gy = np.full_like(h, np.nan), np.full_like(h, np.nan)
    gx[1:-1] = (h[2:] - h[:-2]) / (2 * obs.resolution_m)
    gy[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2 * obs.resolution_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope[~valid] = np.nan
    roughness, discontinuity = _local_stats(h, valid)
    obstacle = valid & (relative > cfg.ordinary_step_m) & (discontinuity > cfg.max_discontinuity_m)
    transition = valid & (discontinuity > cfg.max_discontinuity_m)
    gap = obs.known_unsupported.copy()
    unknown = ~valid & ~gap
    high_risk = unknown | gap | obstacle | transition | (slope > cfg.max_slope_deg) | (roughness > cfg.max_roughness_m)
    traversable = valid & ~high_risk
    foothold = traversable.copy()
    clearance = cfg.foot_clearance_cells
    for dr in range(-clearance, clearance + 1):
        for dc in range(-clearance, clearance + 1):
            shifted = np.roll(high_risk, (dr, dc), axis=(0, 1))
            if dr > 0: shifted[:dr] = True
            if dr < 0: shifted[dr:] = True
            if dc > 0: shifted[:, :dc] = True
            if dc < 0: shifted[:, dc:] = True
            foothold &= ~shifted
    potentially_crossable = obstacle & (relative <= cfg.candidate_step_m)
    return AffordanceMap(slope, roughness, discontinuity, relative, traversable,
                         valid & ~traversable, foothold, high_risk, obstacle,
                         transition, gap, potentially_crossable, unknown, baseline)
