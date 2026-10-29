# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2RoughEnvCfg
from .flat_env_cfg import UnitreeGo2FlatEnvCfg

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from isaaclab.managers import SceneEntityCfg

@configclass
class UnitreeGo2JumpEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # jump task only
        self.events.reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # override rewards
        self.episode_length_s = 20.0
        self.rewards.flat_orientation_l2.weight = -2.5 #2.5
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.jump_height = RewTerm(
            func=mdp.jump_height,
            weight=1.0,
            params={
                "target_height":0.45,
            },
        )
        # jump reward
        self.rewards.jump_airborne = RewTerm(
            func=mdp.jump_airborne,
            weight=0.2,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[".*foot"],
                ),
            },
        )
        self.rewards.jump_takeoff = RewTerm(
            func=mdp.jump_takeoff,
            weight=2.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[".*_foot"],
                ),
            },
        )
        

        #Xtrack velocity rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class UnitreeGo2JumpEnvCfg_PLAY(UnitreeGo2JumpEnvCfg):
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
