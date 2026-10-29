"""One-frame observation -> geometry -> reasoning with measured module latency."""
from __future__ import annotations
from dataclasses import fields
from time import perf_counter
import numpy as np
from cra373.affordance.terrain_observation import TerrainObservation
from cra373.affordance.terrain_affordance import CapabilityHypotheses, estimate_affordance
from cra373.affordance.affordance_reasoner import reason

def analyze(obs: TerrainObservation, cfg: CapabilityHypotheses = CapabilityHypotheses()) -> dict:
    t0 = perf_counter()
    aff = estimate_affordance(obs, cfg)
    t1 = perf_counter()
    result = reason(obs, aff, cfg)
    t2 = perf_counter()
    maps = {f.name: getattr(aff, f.name).astype(int).tolist()
            for f in fields(aff) if isinstance(getattr(aff, f.name), np.ndarray)
            and getattr(aff, f.name).dtype == bool}
    return {
        "source": obs.source, "metadata": obs.metadata, "resolution_m": obs.resolution_m,
        "timestamp_s": obs.timestamp_s, "coordinate_frame": "base_yaw_aligned_world_up",
        "robot_state": {"base_position_w_m": obs.base_position_w_m,
                        "base_yaw_rad": obs.base_yaw_rad,
                        "base_velocity_b_mps": obs.base_velocity_b_mps},
        "height_m": [[float(v) if np.isfinite(v) else None for v in row] for row in obs.height_m],
        "valid": obs.valid.astype(int).tolist(), "maps": maps,
        "slope_deg": np.nan_to_num(aff.slope_deg, nan=-1).tolist(),
        "roughness_m": np.nan_to_num(aff.roughness_m, nan=-1).tolist(),
        "reasoning": result.to_dict(),
        "latency_ms": {"affordance": (t1-t0)*1000, "reasoning": (t2-t1)*1000,
                       "total": (t2-t0)*1000},
        "las": {"status": "LAS model unavailable — training required",
                "latent_vector": None, "reconstruction_mse": None},
    }
