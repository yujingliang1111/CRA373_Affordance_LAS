from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import *


#可成功但亂跑
def forward_progress(
    env,
    asset_cfg=SceneEntityCfg("robot"),
):

    asset = env.scene[asset_cfg.name]

    vx = asset.data.root_lin_vel_w[:,0]

    return torch.clamp(vx, min=0.0)



#success
# def distance_to_goal(
#     env,
#     target_x,
#     asset_cfg=SceneEntityCfg("robot"),
# ):

#     asset = env.scene[asset_cfg.name]

#     robot_x = asset.data.root_pos_w[:,0]
#     origin_x = env.scene.env_origins[:,0]

#     distance = robot_x - origin_x

#     reward = torch.clamp(
#         distance / target_x,
#         0,
#         1
#     )

#     return reward


def distance_to_goal(
    env,
    target_x,
    asset_cfg=SceneEntityCfg("robot"),
):

    asset = env.scene[asset_cfg.name]

    robot_xy = asset.data.root_pos_w[:, :2]
    origin_xy = env.scene.env_origins[:, :2]

    relative_xy = robot_xy - origin_xy

    #reward = relative_xy[:, 0]#**2 + relative_xy[:, 1]**2
    relative_x = relative_xy[:, 0]
    
    reward = torch.clamp(
        relative_x / target_x,
        0,
        1
    )
    return reward


# def foot_edge_penalty(
#     env,
#     radius: float = 0.15,
#     edge_threshold: float = 0.03,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ):
#     """
#     Penalize feet that are close to terrain edges.

#     For each foot:
#         1. Find all height-scanner rays within `radius`.
#         2. Compute the local height difference.
#         3. Penalize if the height difference exceeds edge_threshold.

#     Returns:
#         penalty (num_env,)
#     """


#     print("foot_edge_penalty called")
#     # Robot
#     robot = env.scene[asset_cfg.name]

#     # Height scanner
#     height_scanner = env.scene.sensors["height_scanner"]

#     # -------------------------------------------------------
#     # Get foot positions
#     # shape = (num_env, num_feet, 3)
#     # -------------------------------------------------------

#     foot_ids = robot.find_bodies(".*_foot")[0]
#     foot_pos = robot.data.body_pos_w[:, foot_ids, :]

#     # -------------------------------------------------------
#     # Get ray hit positions
#     # shape = (num_env, num_ray, 3)
#     # -------------------------------------------------------

#     ray_pos = height_scanner.data.ray_hits_w

#     # xyz
#     ray_xy = ray_pos[:, :, :2]
#     ray_z = ray_pos[:, :, 2]

#     num_env = ray_pos.shape[0]
#     num_feet = foot_pos.shape[1]

#     penalty = torch.zeros(num_env, device=ray_pos.device)

#     # -------------------------------------------------------
#     # Check every foot
#     # -------------------------------------------------------

#     for i in range(num_feet):

#         # Foot xy
#         foot_xy = foot_pos[:, i, :2]

#         # ---------------------------------------------
#         # Distance from every ray to current foot
#         #
#         # shape:
#         # (num_env, num_ray)
#         # ---------------------------------------------

#         dist = torch.norm(
#             ray_xy - foot_xy.unsqueeze(1),
#             dim=-1,
#         )

#         # Rays near this foot
#         nearby = dist < radius

#         # ------------------------------------------------
#         # Compute local height difference
#         # ------------------------------------------------

#         local_max = torch.where(
#             nearby,
#             ray_z,
#             torch.full_like(ray_z, -1e6),
#         ).max(dim=1).values

#         local_min = torch.where(
#             nearby,
#             ray_z,
#             torch.full_like(ray_z, 1e6),
#         ).min(dim=1).values

#         local_height_diff = local_max - local_min

#         # ------------------------------------------------
#         # Penalty only if height difference is large
#         # ------------------------------------------------

#         penalty += torch.clamp(
#             local_height_diff - edge_threshold,
#             min=0.0,
#         )

#     # Average over four feet
#     penalty /= num_feet
#     print(local_height_diff.max())

#     # Reward function returns penalty.
#     # Remember to use a NEGATIVE weight.
#     return penalty


def foot_edge_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    radius: float = 0.15,
    edge_threshold: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Penalize feet that contact terrain edges.

    Only feet currently touching the ground are considered.
    """

    robot = env.scene[asset_cfg.name]

    # Contact sensor
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )

    # Height scanner
    height_scanner = env.scene.sensors["height_scanner"]

    ray_pos = height_scanner.data.ray_hits_w

    ray_xy = ray_pos[:, :, :2]
    ray_z = ray_pos[:, :, 2]

    foot_pos = robot.data.body_pos_w[:, sensor_cfg.body_ids, :]

    num_env = ray_pos.shape[0]
    num_feet = len(sensor_cfg.body_ids)

    penalty = torch.zeros(num_env, device=ray_pos.device)

    for i in range(num_feet):

        # 沒接觸就跳過
        if not torch.any(contacts[:, i]):
            continue

        foot_xy = foot_pos[:, i, :2]

        dist = torch.norm(
            ray_xy - foot_xy.unsqueeze(1),
            dim=-1,
        )

        nearby = dist < radius

        local_max = torch.where(
            nearby,
            ray_z,
            torch.full_like(ray_z, -1e6),
        ).max(dim=1).values

        local_min = torch.where(
            nearby,
            ray_z,
            torch.full_like(ray_z, 1e6),
        ).min(dim=1).values

        local_diff = local_max - local_min

        edge_penalty = torch.clamp(
            local_diff - edge_threshold,
            min=0.0,
        )

        # 只有接觸的腳才加進去
        penalty += edge_penalty * contacts[:, i].float()

    return penalty / num_feet

def calf_contact_penalty(
    env,
    sensor_cfg,
):

    contact_sensor = env.scene.sensors[sensor_cfg.name]

    contacts = (
        contact_sensor.data.net_forces_w_history[
            :, :, sensor_cfg.body_ids, :
        ]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )

    return contacts.float().sum(dim=1)