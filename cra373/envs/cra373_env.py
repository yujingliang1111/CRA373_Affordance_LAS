# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors import Camera

from .cra373_env_cfg import CRA373FlatEnvCfg, CRA373RoughEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils


class CRA373Env(DirectRLEnv):
    cfg: CRA373FlatEnvCfg | CRA373RoughEnvCfg

    def __init__(self, cfg: CRA373FlatEnvCfg | CRA373RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*thigh")

        

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # self._front_camera = Camera(self.cfg.front_camera)
        # self.scene.sensors["front_camera"] = self._front_camera

        #if isinstance(self.cfg, CRA373RoughEnvCfg):
            # we add a height scanner for perceptive locomotion
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        if isinstance(self.cfg, CRA373RoughEnvCfg) and self.cfg.visualize_pointcloud:
            self._pointcloud_visualizer = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/PointCloud",
                    markers={
                        "point": sim_utils.SphereCfg(
                            radius=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(),
                        ),
                    },
                )
            )

        # if self.cfg.visualize_pointcloud:
        #     self._pointcloud_visualizer = VisualizationMarkers(
        #         VisualizationMarkersCfg(
        #             prim_path="/Visuals/PointCloud",
        #             markers={
        #                 "point": sim_utils.SphereCfg(
        #                     radius=0.02,
        #                     visual_material=sim_utils.PreviewSurfaceCfg(),
        #                 ),
        #             },
        #         )
        #     )
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos.torch
        if (
            isinstance(self.cfg, CRA373RoughEnvCfg)
            and self.cfg.visualize_pointcloud
        ):
            self._visualize_pointcloud()

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, CRA373RoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w.torch[:, 2].unsqueeze(1)
                - self._height_scanner.data.ray_hits_w.torch[..., 2]
                - 0.5
            ).clip(-1.0, 1.0)
       
        #depth = self.scene["front_camera"].data.output["distance_to_image_plane"]
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b.torch,
                    self._robot.data.root_ang_vel_b.torch,
                    self._robot.data.projected_gravity_b.torch,
                    self._commands,
                    self._robot.data.joint_pos.torch - self._robot.data.default_joint_pos.torch,
                    self._robot.data.joint_vel.torch,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        #observations = {"policy": obs}
        observations = {
            "policy": obs,
            #"depth": depth,
        }
        # print("Depth shape:", depth.shape)
        return observations

    # def _visualize_pointcloud(self):

    #     points_w = self._height_scanner.data.ray_hits_w

    #     # Only visualize environment 0
    #     points = points_w[0]

    #     # One marker type: "point"
    #     marker_indices = torch.zeros(
    #         points.shape[0],
    #         dtype=torch.long,
    #         device=points.device,
    #     )

    #     self._pointcloud_visualizer.visualize(
    #         points,
    #         marker_indices=marker_indices,
    #     )
    def _visualize_pointcloud(self):

        points_w = self._height_scanner.data.ray_hits_w.torch
        
        # print("================================")
        # print("ray_hits_w shape:", points_w.shape)
        # print("ray_hits_w[0, :5]:")
        # print(points_w[0, :5])
        # print("robot pos:")
        # print(self._robot.data.root_pos_w[0])
        # print("================================")

        #points = points_w[0]
        # points = points_w[0:5]

        # valid = torch.isfinite(points).all(dim=-1)
        # valid &= points[:, 2] > -1.0

        # points = points[valid]

        # if points.shape[0] == 0:
        #     return

        # marker_indices = torch.zeros(
        #     points.shape[0],
        #     dtype=torch.long,
        #     device=points.device,
        # )

        # self._pointcloud_visualizer.visualize(
        #     points,
        #     marker_indices=marker_indices,
        # )
        points = points_w[0]

        valid = torch.isfinite(points).all(dim=-1)
        points = points[valid]

        marker_indices = torch.zeros(
            points.shape[0],
            dtype=torch.long,
            device=points.device,
        )

        self._pointcloud_visualizer.visualize(
            translations=points,
            marker_indices=marker_indices,
        )

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b.torch[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b.torch[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b.torch[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b.torch[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque.torch), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc.torch), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt).torch[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time.torch[:, self._feet_ids]
        air_time = torch.sum(torch.clamp(last_air_time - 0.15, min=0.0) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history.torch
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b.torch[:, :2]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history.torch
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES).to(dtype=torch.long)
        elif isinstance(env_ids, wp.array):
            env_ids = wp.to_torch(env_ids).to(device=self.device, dtype=torch.long)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel.torch[env_ids].clone()
        default_root_state = self._robot.data.default_root_state.torch[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
