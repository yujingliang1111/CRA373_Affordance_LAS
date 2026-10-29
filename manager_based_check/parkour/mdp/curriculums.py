"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv




# def jump_gap_curriculum(
#     env,
#     env_ids,
#     asset_cfg=SceneEntityCfg("robot"),
# ):

#     asset = env.scene[asset_cfg.name]

#     terrain = env.scene.terrain

#     #print("jump_gap_curriculum running")


#     # robot distance
#     distance = torch.norm(
#         asset.data.root_pos_w[env_ids,:2]
#         -
#         env.scene.env_origins[env_ids,:2],
#         dim=1
#     )


#     # success condition
#     success = distance > 2.5


#     # failure condition
#     fail = distance < 0.5


#     terrain.update_env_origins(
#         env_ids,
#         move_up=success,
#         move_down=fail
#     )


#     return torch.mean(
#         terrain.terrain_levels.float()
#     )


def jump_gap_curriculum(
    env,
    env_ids,
):

    terrain = env.scene.terrain

    success = env.termination_manager.get_term("success")[env_ids]

    #fail = env.termination_manager.get_term("base_height")[env_ids]
    fail = (
        env.termination_manager.get_term("base_height")[env_ids]
        |
        env.termination_manager.get_term("base_contact")[env_ids]
    )

    terrain.update_env_origins(
        env_ids,
        move_up=success,
        move_down=fail,
    )

    # print(
    #     "success:",
    #     success.sum(),
    #     "fail:",
    #     fail.sum()
    # )

    return torch.mean(
        terrain.terrain_levels.float()
    )