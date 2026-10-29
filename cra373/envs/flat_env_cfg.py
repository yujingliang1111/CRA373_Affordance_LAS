# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import CRA373RoughEnvCfg
from ..CRA373 import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg as VelocityRewardsCfg
from isaaclab.managers import SceneEntityCfg

@configclass
class RewardsCfg(VelocityRewardsCfg):
    forward_progress = RewTerm(
        func=mdp.forward_progress,
        weight=0.0,
    )

    base_height = RewTerm(
        func=mdp.base_height_exp,
        weight=0.0,
        params={
            "target_height": 0.28,
            "std": 0.05
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_calf"],
            ),
            "threshold": 0.1,
            "command_name": "base_velocity",
        },
    )

    calf_contact_penalty = RewTerm(
        func=mdp.calf_contact_penalty,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_calf"]
            ),
        },
    )


@configclass
class CRA373FlatEnvCfg(CRA373RoughEnvCfg):

    rewards: RewardsCfg = RewardsCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25#0.25
        # self.rewards.feet_air_time = RewTerm(
        #     func=mdp.feet_air_time,
        #     weight=0.8,    
        #     params={
        #         "sensor_cfg": SceneEntityCfg(
        #             "contact_forces",
        #             body_names=[".*_foot"],
        #         ),
        #         "threshold": 0.1,
        #         "command_name": "base_velocity",
        #     },
        # )
        self.rewards.base_height.weight = 0.25
        self.rewards.forward_progress.weight = 1.0
        self.rewards.track_lin_vel_xy_exp.weight = 2.5#1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        self.rewards.calf_contact_penalty.weight = -1.0

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class CRA373FlatEnvCfg_PLAY(CRA373FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        
