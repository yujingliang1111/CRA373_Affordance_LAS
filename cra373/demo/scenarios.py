"""Fixed synthetic geometry for reproducible pipeline checks, not sensor recordings."""
from __future__ import annotations
import numpy as np
from cra373.affordance.terrain_observation import TerrainObservation

NAMES = ("Flat", "Rough", "Raised step", "Known gap")

def load_scenario(name: str) -> TerrainObservation:
    if name not in NAMES:
        raise ValueError(f"unknown scenario: {name}")
    rng = np.random.default_rng(373)
    height = np.zeros((17, 11), dtype=float)
    valid = np.ones_like(height, dtype=bool)
    unsupported = np.zeros_like(valid)
    if name == "Rough":
        height += rng.normal(0.0, 0.018, height.shape)
    elif name == "Raised step":
        height[10:] = 0.12
    elif name == "Known gap":
        valid[10:13, 2:9] = False
        unsupported[10:13, 2:9] = True
        height[~valid] = np.nan
    return TerrainObservation(height, valid, 0.1, known_unsupported=unsupported,
                              source="synthetic_offline", metadata={"scenario": name, "seed": 373,
                              "support_truth": name == "Known gap"})
