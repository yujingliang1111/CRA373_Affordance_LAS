# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


# def gap_crossed(
#     env,
#     target_x: float,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Return True when the robot has crossed the gap."""

#     asset = env.scene[asset_cfg.name]

#     return asset.data.root_pos_w[:, 0] > target_x

# def gap_crossed(
#     env,
#     target_x: float,
#     asset_cfg=SceneEntityCfg("robot"),
# ):

#     asset = env.scene[asset_cfg.name]

#     robot_x = asset.data.root_pos_w[:,0]

#     origin_x = env.scene.env_origins[:,0]

#     relative_x = robot_x - origin_x

#     return relative_x > target_x
    # return (
    #     (relative_x > platform_length + gap_width)
    #     &
    #     (asset.data.root_pos_w[:,2] > 0.2)
    # )


def gap_crossed(
    env,
    target_x: float,
    asset_cfg=SceneEntityCfg("robot"),
):

    asset = env.scene[asset_cfg.name]

    robot_xy = asset.data.root_pos_w[:, :2]
    origin_xy = env.scene.env_origins[:, :2]

    relative_xy = robot_xy - origin_xy

    distance = torch.norm(relative_xy, dim=1)

    relative_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

    relative_x = relative_xy[:, 0]
    
    return relative_x > target_x

    # return (
    #     relative_xy[:, 0]**2 +
    #     relative_xy[:, 1]**2
    # ) > target_x**2


def root_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if the robot base is lower than the minimum height."""

    asset: RigidObject = env.scene[asset_cfg.name]

    return asset.data.root_pos_w[:, 2] < minimum_height