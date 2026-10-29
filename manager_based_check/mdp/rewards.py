# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # contact_sensor = env.scene["contact_forces"]

    # print("BODY NAMES:")
    # print(contact_sensor.body_names)

    # print("SHAPE:")
    # print(contact_sensor.data.net_forces_w.shape)

    # print("ENV 0:")
    # for i, name in enumerate(contact_sensor.body_names):
    #     force = contact_sensor.data.net_forces_w[0, i]
    #     print(f"{i}: {name} -> {force}")
    # #no reward for zero command
    # print("first_contact =", first_contact[0])
    # print("last_air_time =", last_air_time[0])
    # print("net_forces =", contact_sensor.data.net_forces_w[0, sensor_cfg.body_ids])

    # print("sensor_cfg.body_ids =", sensor_cfg.body_ids)
    # print("sensor_cfg.body_names =", sensor_cfg.body_names)     
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


# def base_height_exp(
#     env,
#     target_height: float,
#     std: float,
#     asset_cfg=SceneEntityCfg("robot"),
# ):
#     asset = env.scene[asset_cfg.name]

#     height = asset.data.root_pos_w[:, 2]

#     error = torch.square(height - target_height)

#     return torch.exp(-error / std**2)

# def jump_height(
#     env,
#     target_height: float,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ):

#     asset: Articulation = env.scene[asset_cfg.name]

#     base_height = asset.data.root_pos_w[:, 2]

    

#     # reward = torch.exp(
#     #     -((base_height - target_height) ** 2) / 0.05
#     # )
#     reward = torch.clamp(
#         (base_height - 0.3) / (target_height - 0.3),
#         0,
#         1,
#     )

#     reward = reward + torch.where(
#         base_height > target_height,
#         torch.ones_like(base_height) * 1000,
#         torch.zeros_like(base_height),
#     )

#     return reward






def jump_takeoff(
    env,
    sensor_cfg: SceneEntityCfg,
):

    asset = env.scene["robot"]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    # 四腳離地
    airborne = torch.all(
        air_time > 0.02,
        dim=1
    )

    # 向上速度
    vz = asset.data.root_lin_vel_w[:, 2]

    reward = (
        airborne &
        (vz > 0.2)
    ).float()

    return reward

def jump_height(
    env,
    target_height: float,
    asset_cfg=SceneEntityCfg("robot"),
):

    asset = env.scene[asset_cfg.name]

    height = asset.data.root_pos_w[:,2]

    reward = torch.exp(
        -((height - target_height)**2) / 0.005
    )

    return reward

def jump_airborne(
    env,
    sensor_cfg,
):

    contact_sensor = env.scene.sensors[sensor_cfg.name]

    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    reward = torch.mean(air_time, dim=1)

    return torch.clamp(reward, max=0.2)