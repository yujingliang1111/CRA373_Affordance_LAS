# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .gap_terrain import gap_terrain

from isaaclab.terrains import mesh_terrains
from .random_platform_terrain import random_platform_terrain
from .random_stones_terrains import my_stepping_stones_terrain


@configclass
class GapTerrainCfg(SubTerrainBaseCfg):
    """Simple fixed gap terrain."""

    function = gap_terrain

    # 左右平台長度
    platform_length: float = 2.0

    # 平台寬度
    platform_width: float = 3.0

    # gap 寬度
    gap_width_range = (0.05, 0.20)

    # 平台高度
    platform_height: float = 0.15
    

@configclass
class RandomPlatformTerrainCfg(SubTerrainBaseCfg):

    function = random_platform_terrain

    platform_length_range = (0.6,2.0)

    gap_width_range = (0.1,0.8)

    platform_width = 3.0

    platform_height = 0.15

    num_platforms = 8



ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""




GAP_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0,8.0),
    border_width=1.0,

    num_rows=5,
    num_cols=5,

    horizontal_scale=0.1,
    vertical_scale=0.005,

    use_cache=False,

    curriculum=True,

    sub_terrains={
        "gap": GapTerrainCfg(
            proportion=1.0,
            platform_length=2.0,
            platform_width=3.0,
            gap_width_range=(0.05, 0.7),
            #gap_width = terrain.cfg.sub_terrains["gap"].gap_width,
            platform_height=0.15,
        )
    },
)


STAIRS_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)


BOX_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)


MIXED_PARKOUR_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)

RANDOM_PLATFORM_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 4.0),
    border_width=1.0,

    num_rows=5,
    num_cols=5,

    horizontal_scale=0.1,
    vertical_scale=0.005,

    use_cache=False,

    curriculum=True,

    sub_terrains={
        "random_platform": RandomPlatformTerrainCfg(
            proportion=1.0,
            platform_length_range=(1.8, 1.8),
            gap_width_range=(0.0, 0.6),
            platform_width=3.0,
            platform_height=0.15,
            num_platforms=5,
        )
    },
)

STONES_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,

    num_rows=10,
    num_cols=20,

    horizontal_scale=0.1,
    vertical_scale=0.005,

    slope_threshold=0.75,

    use_cache=False,
    curriculum=True,

    sub_terrains={
        
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.2,
        ),


        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(

            proportion=0.8,

            # 石頭高度(最大起伏)
            stone_height_max=0.05,

            # 石頭大小(m)
            stone_width_range=(0.4, 0.7),#(0.5, 0.8)

            # 石頭間距(m)
            stone_distance_range=(0.0, 0.15),#0.35

            # 洞深(m)
            holes_depth=-2.0,

            # 中央生成平台
            platform_width=2.0,
        ),

    },
)


MY_STONE_CFG = TerrainGeneratorCfg(
    size=(8.0,8.0),

    horizontal_scale=0.1,
    vertical_scale=0.005,

    num_rows=10,
    num_cols=20,

    # num_rows=1,
    # num_cols=5,

    curriculum=True,

    sub_terrains={
        "stones": terrain_gen.HfTerrainBaseCfg(
            function=my_stepping_stones_terrain,
            proportion=1.0,
        )
    }
)