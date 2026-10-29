import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


ROBOT_USD_PATH = Path(__file__).resolve().parents[2] / "CRA373_model" / "cra373.usd"

CRA373_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.32),
        rot=(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": -20.0 * math.pi / 180.0,
            ".*_calf_joint": 25.0 * math.pi / 180.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,

    actuators={
        "base_link": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_joint",
                ".*_thigh_joint",
                ".*_calf_joint",
            ],

            effort_limit_sim=30.0,
            velocity_limit_sim=20.0,

            stiffness=25.0,
            damping=0.5,
        ),
    },
    
)
"""CRA373 robot configuration using Isaac Lab's built-in implicit actuator."""
