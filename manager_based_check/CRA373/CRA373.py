
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.managers import EventTermCfg as EventTerm
import math

CRA373_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/isaaclab/CRA373_isaacsim/cra373.usd",#/CRA373_isaacsim/cra373.usd", #{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",/CRA373_URDF_delivery/cra373_edited.usd
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
        rot=(math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": -20.0 * math.pi / 180.0,
            ".*_calf_joint": 25.0 * math.pi / 180.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=40.0,#23.5,
            saturation_effort=40.0,#23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
    
)
"""Configuration of ITRI Dog using DC-Motor actuator model."""
