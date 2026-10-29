# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg as VelocityRewardsCfg

##
# Pre-defined configs
##
from .CRA373 import CRA373_CFG  # isort: skip
# from .parkour_terrain_cfg import GAP_TERRAIN_CFG

from ...parkour import mdp

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg



from ...parkour.terrain.parkour_terrain_cfg import (
    ROUGH_TERRAINS_CFG,
    GAP_TERRAIN_CFG,
    STAIRS_TERRAIN_CFG,
    BOX_TERRAIN_CFG,
    MIXED_PARKOUR_CFG,
    RANDOM_PLATFORM_TERRAIN_CFG,
    STONES_TERRAINS_CFG,
    MY_STONE_CFG,
)




#termination term
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab_tasks.manager_based.locomotion.velocity.parkour.mdp import terminations as parkour_mdp

# =========================================
# Parkour Training Setting
# =========================================

@configclass
class ParkourTrainingCfg:
    """Training scenario configuration."""

    # primitive skill
    skill: str = "my_stepping_stones"#"stepping_stones"#"my_stepping_stones"#"stepping_stones"#"random_platform" #"gap"


@configclass
class CurriculumCfg:
    jump_gap_levels = CurrTerm(func=mdp.jump_gap_curriculum)


@configclass
class RewardsCfg(VelocityRewardsCfg):

    distance_to_goal = RewTerm(
        func=mdp.distance_to_goal,
        weight=0.0,
        params={
            "target_x": 7.0,
        },
    )

    forward_progress = RewTerm(
        func=mdp.forward_progress,
        weight=0.0,
    )

    foot_edge_penalty = RewTerm(
        func=mdp.foot_edge_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*_foot",
            ),
            "radius": 0.15,
            "edge_threshold": 0.03,
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


# =========================================
# Parkour Environment
# =========================================

@configclass
class CRA373ParkourEnvCfg(LocomotionVelocityRoughEnvCfg):

    training: ParkourTrainingCfg = ParkourTrainingCfg()
    
    curriculum: CurriculumCfg = CurriculumCfg()

    rewards: RewardsCfg = RewardsCfg()

    
    def __post_init__(self):
        
        # post init of parent
        super().__post_init__()

        
        #robot
        self.scene.robot = CRA373_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        #self.scene.height_scanner.debug_vis = True
        
        # Parkour Skill Setting
        self.setup_skill()


        # reduce action scale
        # Action
        self.actions.joint_pos.scale = 0.25


        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        #command
        self.commands.base_velocity.ranges.lin_vel_x = (1.0,1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0,0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0,0.0)

        self.commands.base_velocity.heading_command = False

        #curriculum
        #self.curriculum.terrain_levels = None


        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_acc_l2.weight = -2.5e-7
        

        


        
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"
        # Fail if robot falls
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={
                "minimum_height": 0.08,
            },
        )

        # Success if robot crosses the gap
        self.terminations.success = DoneTerm(
            func=parkour_mdp.gap_crossed,
            params={
                "target_x": 7,#4
            },
        )
    
    # ==================================================
    # Skill configuration
    # ==================================================

    def setup_skill(self):

        skill = self.training.skill

        # Gap
        if skill == "gap":

            self.scene.terrain.terrain_type = "generator"
            self.scene.terrain.terrain_generator = GAP_TERRAIN_CFG
            #self.scene.terrain.terrain_generator.curriculum = True

            self.episode_length_s = 5.0


            # disable locomotion reward
            self.rewards.track_lin_vel_xy_exp.weight = 0.0
            self.rewards.track_ang_vel_z_exp.weight = 0.0


            # enable parkour reward
            self.rewards.flat_orientation_l2.weight = -1.0
            self.rewards.feet_air_time.weight = 0.05


            #self.rewards.distance_to_goal.weight = 3.0
            # self.rewards.distance_to_goal = RewTerm(
            #     func=mdp.distance_to_goal,
            #     weight=10.0,
            #     params={
            #         "target_x": 1.5,
            #         #"std": 0.5,
            #     },
            # )


            # #self.rewards.forward_progress.weight = 2.0
            # self.rewards.forward_progress = RewTerm(
            #     func=mdp.forward_progress,
            #     weight=5.0,
            #     params={
            #         #"target_x":1.5,
            #     },
            # )

        
        # Step Up
        elif skill == "step_up":

            self.scene.terrain.terrain_type = "generator"

            self.scene.terrain.terrain_generator = BOX_TERRAIN_CFG
            
            self.episode_length_s = 4.0

        
        # Stairs
        elif skill == "stairs":

            self.scene.terrain.terrain_type = "generator"

            self.scene.terrain.terrain_generator = STAIRS_TERRAIN_CFG
            self.episode_length_s = 6.0


        # Mixed Parkour
        elif skill == "mixed":

            self.scene.terrain.terrain_type = "generator"

            self.scene.terrain.terrain_generator = MIXED_PARKOUR_CFG
            self.episode_length_s = 10.0

        # Random Platform
        elif skill == "random_platform":

            self.scene.terrain.terrain_type = "generator"

            self.scene.terrain.terrain_generator = RANDOM_PLATFORM_TERRAIN_CFG

            self.rewards.distance_to_goal.weight = 10.0
            self.rewards.forward_progress.weight = 5.0

            self.episode_length_s = 5.0


            # disable locomotion reward
            self.rewards.track_lin_vel_xy_exp.weight = 0.0
            self.rewards.track_ang_vel_z_exp.weight = 0.0


            # enable parkour reward
            self.rewards.flat_orientation_l2.weight = -1.0
            self.rewards.feet_air_time.weight = 0.05


        # Random Platform
        elif skill == "stepping_stones":

            self.scene.terrain.terrain_type = "generator"

            self.scene.terrain.terrain_generator = STONES_TERRAINS_CFG

            self.rewards.distance_to_goal.weight = 0.0
            self.rewards.forward_progress.weight = 2.0

            self.episode_length_s = 10.0


            # disable locomotion reward
            self.rewards.track_lin_vel_xy_exp.weight = 2.0
            self.rewards.track_ang_vel_z_exp.weight = 0.0


            # enable parkour reward
            self.rewards.flat_orientation_l2.weight = -1.0
            self.rewards.feet_air_time.weight = 0.05
            self.rewards.foot_edge_penalty.weight = -5.0

        elif skill == "my_stepping_stones":

            self.scene.terrain.terrain_type = "generator"

            self.scene.terrain.terrain_generator = MY_STONE_CFG

            self.rewards.distance_to_goal.weight = 3.0
            self.rewards.forward_progress.weight = 2.0

            self.episode_length_s = 20.0


            # disable locomotion reward
            self.rewards.track_lin_vel_xy_exp.weight = 0.5 #1.0
            self.rewards.track_ang_vel_z_exp.weight = 0.0


            # enable parkour reward
            self.rewards.flat_orientation_l2.weight = -1.0
            self.rewards.feet_air_time.weight = 0.05
            self.rewards.foot_edge_penalty.weight = -5.0
            self.rewards.calf_contact_penalty.weight = -1.0



        else:
            raise ValueError(
                f"Unknown parkour skill: {skill}"
            )



# Play Config
@configclass
class CRA373ParkourEnvCfg_PLAY(CRA373ParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        #command
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        
