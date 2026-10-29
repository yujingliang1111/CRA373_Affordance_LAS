"""Deterministic action feasibility reasoning; scores are uncalibrated heuristics."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from .terrain_observation import TerrainObservation
from .terrain_affordance import AffordanceMap, CapabilityHypotheses
from .robot_capability import read_robot_capability

@dataclass
class ActionAssessment:
    action: str
    heuristic_score: float | None
    execution_supported: bool
    reason: str

@dataclass
class AffordanceReasoningResult:
    features: dict
    actions: list[ActionAssessment]
    selected_action: str
    explanation: str
    uncertainty: list[str]
    capability_constraints: dict

    def to_dict(self) -> dict:
        return dict(features=self.features, actions=[asdict(a) for a in self.actions],
                    selected_action=self.selected_action, explanation=self.explanation,
                    uncertainty=self.uncertainty, capability_constraints=self.capability_constraints)

def reason(obs: TerrainObservation, aff: AffordanceMap,
           cfg: CapabilityHypotheses = CapabilityHypotheses()) -> AffordanceReasoningResult:
    x, y = obs.xy_grid_m
    ahead = (x > 0) & (np.abs(y) <= 0.20)
    if not ahead.any():
        ahead = x >= 0
    n = int(ahead.sum())
    valid_ahead = ahead & obs.valid
    valid_fraction = float(valid_ahead.sum() / n)
    safe_fraction = float((ahead & aff.traversable).sum() / n)
    gap_fraction = float((ahead & aff.gap).sum() / n)
    unknown_fraction = float((ahead & aff.unknown).sum() / n)
    heights = aff.relative_height_m[valid_ahead]
    obstacle_height = float(max(0, np.max(heights))) if heights.size else None
    slopes = aff.slope_deg[valid_ahead]
    slopes = slopes[np.isfinite(slopes)]
    slope = float(np.max(slopes)) if slopes.size else None
    rough = aff.roughness_m[valid_ahead]
    roughness = float(np.max(rough)) if rough.size else None
    step = obstacle_height is not None and obstacle_height > cfg.ordinary_step_m
    candidate_step = step and obstacle_height <= cfg.candidate_step_m
    landing_cells = max(1, int(np.ceil(cfg.min_landing_length_m / obs.resolution_m)))
    obstacle_rows = np.where((ahead & aff.obstacle).any(axis=1))[0]
    landing_ok = False
    if obstacle_rows.size:
        start = int(obstacle_rows.max()) + 1
        stop = start + landing_cells
        landing_ok = stop <= obs.height_m.shape[0] and all(
            (ahead[r] & aff.traversable[r]).any() for r in range(start, stop))
    uncertainty = ["Decision thresholds are demo hypotheses; CRA373 physical feasibility is unvalidated."]
    if unknown_fraction:
        uncertainty.append("Missing ray hits are unknown support, not measured gaps.")
    if gap_fraction:
        uncertainty.append("Gap truth comes from the offline scenario, not the height scanner.")
    features = dict(ahead_valid_fraction=valid_fraction, ahead_traversable_fraction=safe_fraction,
                    ahead_unknown_fraction=unknown_fraction, ahead_gap_fraction=gap_fraction,
                    obstacle_height_m=obstacle_height, max_slope_deg=slope,
                    max_roughness_m=roughness, landing_region_sufficient=landing_ok,
                    base_speed_mps=float(np.linalg.norm(obs.base_velocity_b_mps[:2])))
    walk_score = float(np.clip(safe_fraction * (1-gap_fraction-unknown_fraction), 0, 1))
    slow_score = float(np.clip(valid_fraction * (1-gap_fraction) * (0.6+0.4*safe_fraction), 0, 1))
    step_score = 0.65 if candidate_step and landing_ok else 0.0
    avoid_score = float(np.clip(0.3+0.6*(1-safe_fraction), 0, 1))
    stop_score = float(np.clip(0.1+0.9*max(gap_fraction, unknown_fraction, 1-valid_fraction), 0, 1))
    actions = [
        ActionAssessment("Walk", walk_score, True, "Forward cells satisfy configured geometric thresholds."),
        ActionAssessment("Slow Walk", slow_score, True, "Uneven terrain; safe speed is not calibrated."),
        ActionAssessment("Step Over / Climb", step_score, cfg.climb_controller_validated,
                         "Candidate geometry only; no validated climb controller."),
        ActionAssessment("Jump", None, cfg.jump_controller_validated,
                         "No validated jump controller or dynamic feasibility estimate."),
        ActionAssessment("Avoid", avoid_score, False, "Route planning is not connected."),
        ActionAssessment("Stop", stop_score, True, "Conservative fallback for uncertain support."),
    ]
    if gap_fraction > 0 or unknown_fraction > 0.25 or valid_fraction < 0.7:
        selected = "Stop"
    elif candidate_step and landing_ok:
        selected = "Step Over / Climb"
    elif step or safe_fraction < 0.45:
        selected = "Avoid"
    elif safe_fraction < 0.8 or (roughness is not None and roughness > cfg.max_roughness_m / 2):
        selected = "Slow Walk"
    else:
        selected = "Walk"
    explanations = {
        "Walk": "Forward support is mostly low-risk under the configured geometric thresholds.",
        "Slow Walk": "Forward geometry is uneven. Lower speed is a heuristic recommendation.",
        "Step Over / Climb": "A candidate-height step and landing strip are present. Execution needs a validated controller.",
        "Avoid": "Forward geometry exceeds ordinary walking thresholds. An alternative route must be planned.",
        "Stop": "Support ahead is missing or uncertain; the conservative recommendation is to stop.",
    }
    return AffordanceReasoningResult(features, actions, selected, explanations[selected],
                                     uncertainty, {"threshold_hypotheses": asdict(cfg), "robot": read_robot_capability()})
