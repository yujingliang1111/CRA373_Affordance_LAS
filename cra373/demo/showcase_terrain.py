"""Deterministic rough / stair / gap terrain layout for the live Isaac demo.

Imported only after AppLauncher starts. This configures the demo's copy of the
rough environment; training configs and original project assets are untouched.
"""
from __future__ import annotations

import isaaclab.terrains as terrain_gen

FEATURES = ("rough", "stairs", "gap")


def configure_live_terrain(cfg, scene: str) -> None:
    terrain = cfg.terrain.terrain_generator
    if terrain is None:
        raise ValueError("CRA373 rough environment has no terrain generator")
    terrain.seed = 373
    terrain.use_cache = False
    if scene == "rough":
        rough = terrain.sub_terrains["random_rough"].copy()
        rough.proportion = 1.0
        rough.noise_range = (0.02, 0.10)
        rough.noise_step = 0.02
        terrain.sub_terrains = {"random_rough": rough}
        terrain.num_rows = 3
        terrain.num_cols = 3
        terrain.curriculum = False
        cfg.terrain.max_init_terrain_level = 1
        return
    if scene != "showcase":
        raise ValueError(f"unknown terrain scene: {scene}")

    rough = terrain.sub_terrains["random_rough"].copy()
    rough.proportion = 1.0 / 3.0
    rough.noise_range = (0.03, 0.09)
    rough.noise_step = 0.02
    rough.border_width = 0.10

    stairs = terrain.sub_terrains["pyramid_stairs"].copy()
    stairs.proportion = 1.0 / 3.0
    stairs.step_height_range = (0.08, 0.08)
    stairs.step_width = 0.25
    stairs.platform_width = 1.0
    stairs.border_width = 0.25
    stairs.holes = False

    gap = terrain_gen.MeshGapTerrainCfg(
        proportion=1.0 / 3.0,
        gap_width_range=(0.28, 0.28),
        platform_width=1.2,
    )
    terrain.sub_terrains = {"rough": rough, "stairs": stairs, "gap": gap}
    terrain.size = (4.0, 4.0)
    terrain.border_width = 1.0
    terrain.num_rows = 1
    terrain.num_cols = 3
    terrain.curriculum = True  # one deterministic column for each feature
    terrain.difficulty_range = (0.5, 0.5)
    cfg.terrain.max_init_terrain_level = 0


def set_showcase_focus(env, feature: str) -> None:
    """Move the single demo robot's next reset origin to a selected tile."""
    if feature not in FEATURES:
        raise ValueError(f"unknown showcase feature: {feature}")
    importer = env.unwrapped._terrain
    column = FEATURES.index(feature)
    if importer.terrain_origins is None or importer.terrain_origins.shape[:2] != (1, 3):
        raise RuntimeError("showcase terrain did not create three deterministic tiles")
    importer.terrain_levels[0] = 0
    importer.terrain_types[0] = column
    importer.env_origins[0] = importer.terrain_origins[0, column]
