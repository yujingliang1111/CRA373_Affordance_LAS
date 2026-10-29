"""
Copyright (c) 2026, Mekion
Copyright (c) 2022-2026, The Isaac Lab Project Developers.
SPDX-License-Identifier: Apache-2.0

PJI106 / XB42M custom actuator model.
"""

import torch
from isaaclab.actuators.actuator_base import ActuatorBase
from isaaclab.actuators.actuator_cfg import ActuatorBaseCfg
from isaaclab.utils import configclass
import math


class XB42MActuator(ActuatorBase):
    cfg: "XB42MCfg"

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        n = self._num_envs       # 環境數量
        nj = self.num_joints     # 關節數量
        d = self._device         # 計算裝置

        # ============================================================
        # Initial joint position
        # ============================================================

        default_pos = torch.tensor(
            cfg.init_pose_deg,
            device=d,
            dtype=torch.float32
        )
        default_pos = default_pos.unsqueeze(0).expand(n, -1)
        default_pos = torch.deg2rad(default_pos)

        # Servo 內部追蹤位置
        self._servo_tracked = default_pos.clone()

        # Backlash 大小
        self.backlash_rad = torch.zeros(n, nj, device=d)

        # 齒輪實際接觸位置
        self._contact_pos = default_pos.clone()

        # ============================================================
        # Servo dynamics
        # ============================================================

        # 一階 Servo 模型：
        # alpha = 1 - exp(-2*pi*bandwidth*dt)
        #
        # TODO:
        # bandwidth 目前尚未從廠商取得
        alpha_val = 1.0 - math.exp(
            -2.0 * math.pi * cfg.bandwidth_hz * cfg.sim_dt
        )

        self._alpha = torch.full(
            (n, nj),
            alpha_val,
            device=d
        )

        # Position controller Kp / stiffness
        self.stiffness[:] = cfg.stiffness

        # Position controller Kd / damping
        self.damping[:] = cfg.damping

        # ============================================================
        # Friction
        # ============================================================

        # 摩擦參數
        self._friction = torch.full(
            (n, nj),
            cfg.friction_nm,
            device=d
        )

        # 加速度限制（目前未使用）
        self._acc_limit = torch.zeros(
            n,
            nj,
            device=d
        )

        # ============================================================
        # Internal states
        # ============================================================

        # 前一 timestep 的 torque
        self._torque_prev = torch.zeros(
            n,
            nj,
            device=d
        )

        # ============================================================
        # Command delay
        # ============================================================

        # 控制訊號 delay buffer
        max_delay = cfg.bus_delay_steps + 2

        self._cmd_buffer = (
            default_pos
            .clone()
            .unsqueeze(0)
            .expand(max_delay + 1, -1, -1)
            .clone()
        )

        # 每個環境的 delay timestep
        self._delay_offset = torch.full(
            (n,),
            cfg.bus_delay_steps,
            dtype=torch.long,
            device=d
        )

        # Buffer index
        self._buf_idx = 0

        # 最後一次收到的 delayed command
        self._last_delayed_cmd = default_pos.clone()

    def reset(self, env_ids):
        """Reset actuator internal states for given env indices."""

        # IsaacLab 3.0 may provide env_ids as a Warp array.
        # Convert it to a PyTorch tensor before using it for indexing.
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(
                env_ids,
                device=self._device,
                dtype=torch.long,
            )
        else:
            env_ids = env_ids.to(
                device=self._device,
                dtype=torch.long,
            )

        default_pos = torch.tensor(
            self.cfg.init_pose_deg,
            device=self._device,
            dtype=torch.float32,
        )
        default_pos = torch.deg2rad(default_pos)

        # Reset servo tracking position
        self._servo_tracked[env_ids] = default_pos

        # Reset command buffer
        self._cmd_buffer[:, env_ids, :] = default_pos

        # Reset torque
        self._torque_prev[env_ids] = 0.0

        # ============================================================
        # Backlash randomization
        # ============================================================

        low_rad = math.radians(self.cfg.backlash_range_deg[0])
        high_rad = math.radians(self.cfg.backlash_range_deg[1])

        num_reset_envs = env_ids.numel()

        new_backlash = torch.empty(
            num_reset_envs,
            self.num_joints,
            device=self._device,
            dtype=torch.float32,
        ).uniform_(
            low_rad,
            high_rad,
        )

        self.backlash_rad[env_ids] = new_backlash

        # Randomize gear initial contact position
        half_bl = new_backlash / 2.0

        offset = (
            torch.empty_like(half_bl).uniform_(-1.0, 1.0)
            * half_bl
        )

        self._contact_pos[env_ids] = default_pos + offset

        self._last_delayed_cmd[env_ids] = default_pos


    def compute(self, control_action, joint_pos, joint_vel):

        # ============================================================
        # 1. Bus / Communication Delay
        # ============================================================

        self._cmd_buffer[self._buf_idx] = (
            control_action.joint_positions
        )

        env_idx = torch.arange(
            joint_pos.shape[0],
            device=joint_pos.device
        )

        read_idx = (
            self._buf_idx - self._delay_offset
        ) % self._cmd_buffer.shape[0]

        delayed_cmd = self._cmd_buffer[
            read_idx,
            env_idx,
            :
        ]

        self._buf_idx = (
            self._buf_idx + 1
        ) % self._cmd_buffer.shape[0]

        # ============================================================
        # 2. First-order Servo Lag
        # ============================================================

        self._servo_tracked = (
            self._servo_tracked
            + self._alpha
            * (
                delayed_cmd
                - self._servo_tracked
            )
        )

        self._last_delayed_cmd = delayed_cmd.clone()

        # ============================================================
        # 3. Backlash
        # ============================================================

        half_bl = self.backlash_rad / 2.0

        lower_bound = (
            self._servo_tracked
            - half_bl
        )

        upper_bound = (
            self._servo_tracked
            + half_bl
        )

        new_contact_pos = torch.clamp(
            self._contact_pos,
            lower_bound,
            upper_bound
        )

        # 判斷是否重新接觸齒輪
        reengaged = (
            new_contact_pos
            != self._contact_pos
        )

        if reengaged.any():

            # PJI106 backlash 隨機化
            low_rad = math.radians(
                self.cfg.backlash_range_deg[0]
            )

            high_rad = math.radians(
                self.cfg.backlash_range_deg[1]
            )

            new_bl = torch.empty_like(
                self.backlash_rad
            ).uniform_(
                low_rad,
                high_rad
            )

            self.backlash_rad = torch.where(
                reengaged,
                new_bl,
                self.backlash_rad
            )

        self._contact_pos = new_contact_pos

        # ============================================================
        # 4. Position Error
        # ============================================================

        raw_err = (
            self._contact_pos
            - joint_pos
        )

        effective_err = raw_err

        # ============================================================
        # 5. PD Position Controller
        # ============================================================

        # tau = Kp * position_error - Kd * velocity
        tau = (
            self.stiffness * effective_err
            - self.damping * joint_vel
        )

        # ============================================================
        # 6. Friction
        # ============================================================

        # TODO:
        # 目前使用等效摩擦 0.67 Nm。
        # 更完整模型可拆成 Coulomb + viscous friction。
        tau = (
            tau
            - self._friction
            * torch.sign(joint_vel)
        )

                # ============================================================
        # 7. PJI106 Torque-Speed Curve (T-N Curve)
        # ============================================================

        # 將角速度從 rad/s 轉成 RPM
        # RPM = rad/s * 60 / (2*pi)
        rpm = torch.abs(joint_vel) * 60.0 / (2.0 * math.pi)

        # 取得速度區間
        rpm_1 = self.cfg.tn_rpm_1       # 75 RPM
        rpm_2 = self.cfg.tn_rpm_2       # 175 RPM
        rpm_3 = self.cfg.tn_rpm_3       # 200 RPM

        # 取得對應扭矩
        torque_1 = self.cfg.tn_torque_1  # 60 Nm
        torque_2 = self.cfg.tn_torque_2  # 10 Nm
        torque_3 = self.cfg.tn_torque_3  # 0 Nm

        # ------------------------------------------------------------
        # 0 ~ 75 RPM
        # Torque = 60 Nm
        # ------------------------------------------------------------

        tau_max = torch.full_like(
            rpm,
            torque_1
        )

        # ------------------------------------------------------------
        # 75 ~ 175 RPM
        # 60 Nm → 10 Nm
        # 線性下降
        # ------------------------------------------------------------

        mask_1 = (
            (rpm > rpm_1)
            & (rpm < rpm_2)
        )

        tau_75_175 = (
            torque_1
            + (torque_2 - torque_1)
            * (rpm - rpm_1)
            / (rpm_2 - rpm_1)
        )

        tau_max = torch.where(
            mask_1,
            tau_75_175,
            tau_max
        )

        # ------------------------------------------------------------
        # 175 ~ 200 RPM
        # 10 Nm → 0 Nm
        # 線性下降
        # ------------------------------------------------------------

        mask_2 = (
            (rpm >= rpm_2)
            & (rpm < rpm_3)
        )

        tau_175_200 = (
            torque_2
            + (torque_3 - torque_2)
            * (rpm - rpm_2)
            / (rpm_3 - rpm_2)
        )

        tau_max = torch.where(
            mask_2,
            tau_175_200,
            tau_max
        )

        # ------------------------------------------------------------
        # >= 200 RPM
        # Torque = 0 Nm
        # ------------------------------------------------------------

        tau_max = torch.where(
            rpm >= rpm_3,
            torch.zeros_like(rpm),
            tau_max
        )

        # ------------------------------------------------------------
        # Apply symmetric torque limit
        # ------------------------------------------------------------

        tau = torch.clamp(
            tau,
            -tau_max,
            tau_max
        )

        # ============================================================
        # 8. Output
        # ============================================================

        self.computed_effort[:] = tau
        self.applied_effort[:] = tau

        control_action.joint_efforts = tau
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action


@configclass
class XB42MCfg(ActuatorBaseCfg):

    class_type: type = XB42MActuator

    # ================================================================
    # PJI106 / XB42M Basic Parameters
    # ================================================================

    # ------------------------------------------------
    # Electrical parameters
    # ------------------------------------------------

    rated_voltage: float = 48.0
    # V | 額定電壓

    rated_current: float = 5.2
    # A | 額定電流

    peak_current: float = 45.0
    # A | 峰值電流

    torque_constant: float = 2.36
    # Nm/A | 轉矩常數 Kt

    back_emf_constant: float = 17.0
    # V/kRPM | 反電動勢常數 Ke

    resistance: float = 0.39
    # Ohm | 線電阻

    inductance: float = 0.275e-3
    # H | 線電感

    # ------------------------------------------------
    # Mechanical / Output parameters
    # ------------------------------------------------

    gear_ratio: float = 9.0
    # 1:9 | 減速比

    stall_torque_nm: float = 60.0
    # Nm | 模組輸出端峰值扭矩
    # 注意：60 Nm 是驅動器限制後的實際峰值
    # 不是電氣堵轉扭矩 98.2 Nm

    rated_torque_nm: float = 10.0
    # Nm | 連續額定扭矩

    max_velocity: float = math.radians(200.0 * 6.0)
    # rad/s | 模組輸出端空載轉速約 200 RPM
    # 200 RPM = 20.94 rad/s

    rated_velocity: float = math.radians(175.0 * 6.0)
    # rad/s | 模組輸出端額定轉速 175 RPM

    # ------------------------------------------------
    # Friction
    # ------------------------------------------------

    friction_nm: float = 0.67
    # Nm | 等效摩擦基準（Back drivability）
    # 目前先近似為 Coulomb friction

    cogging_torque_nm: float = 0.065
    # Nm | 齒槽效應扭矩

    # ------------------------------------------------
    # Backlash
    # ------------------------------------------------

    backlash_range_deg: tuple = (0.0, 0.25)
    # deg | PJI106 backlash <= 15 arcmin = 0.25 deg
    #
    # TODO:
    # 廠商只提供「<= 15 arcmin」
    # 尚未知道實際平均值 / 分布，因此目前使用 0~0.25 deg

    # ------------------------------------------------
    # Inertia
    # ------------------------------------------------

    # armature: float = None
    # # kg*m^2 | TODO: 馬達轉子慣量 / 等效輸出端慣量
    # # 目前尚未取得

    # # ================================================================
    # # Servo / Position Control
    # # ================================================================

    # stiffness: float = None
    # # Nm/rad | TODO: Position Controller Kp
    # # 尚未取得廠商參數

    # damping: float = None
    # # Nm*s/rad | TODO: Position Controller Kd
    # # 尚未取得廠商參數

    # bandwidth_hz: float = None
    # # Hz | TODO: 實際 Servo bandwidth
    # # 1 kHz 是 control loop frequency，不等於 servo bandwidth

    # bus_delay_steps: int = None
    # # timestep | TODO: 實際 command latency
    # # 目前只知道 control loop = 1 kHz
    # # 尚未確認 CAN + driver + controller 的實際 delay

    armature: float = 0.01
    # kg*m^2
    # 暫時值，代表輸出端等效慣量
    # 中型機器狗關節常見約 0.005 ~ 0.03

    stiffness: float = 120.0
    # Nm/rad
    # Position Kp
    # 先假設比 Bimo 高約 2 倍

    damping: float = 3.0
    # Nm*s/rad
    # Position Kd
    # 約為 stiffness 的 2~3%

    bandwidth_hz: float = 15.0
    # Hz
    # 真實 servo bandwidth 常見：
    # 小伺服：3~8 Hz
    # 機器狗關節：10~30 Hz

    bus_delay_steps: int = 1
    # timestep
    # sim_dt = 0.005
    # delay = 5 ms

    control_frequency_hz: float = 1000.0
    # Hz | 驅動器控制迴路頻率

        # ================================================================
    # PJI106 Torque-Speed Curve
    # ================================================================

    tn_rpm_1: float = 75.0
    # RPM | 0~75 RPM：峰值扭矩平坦區

    tn_torque_1: float = 60.0
    # Nm | 75 RPM 以下最大輸出扭矩

    tn_rpm_2: float = 175.0
    # RPM | 額定工作轉速

    tn_torque_2: float = 10.0
    # Nm | 175 RPM 額定連續扭矩

    tn_rpm_3: float = 200.0
    # RPM | 空載轉速極限

    tn_torque_3: float = 0.0
    # Nm | 200 RPM 時扭矩降至 0

    # ================================================================
    # Thermal limits
    # ================================================================

    thermal_temperature_limit_c: float = 125.0
    # °C | 溫控上限

    thermal_torque_nm: tuple = (30.0, 40.0, 50.0, 60.0)
    # Nm | 高負載熱限制測試點

    thermal_time_sec: tuple = (170.0, 24.0, 10.0, 6.0)
    # sec | 對應高負載可持續時間

    # ================================================================
    # Encoder
    # ================================================================

    motor_encoder_bits: int = None
    # bit | TODO: 馬達端 encoder 實際解析度

    output_encoder_bits: int = None
    # bit | TODO: 減速端 encoder 實際解析度

    # ================================================================
    # Initial pose
    # ================================================================

    init_pose_deg: list = [
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    ]
    # Degrees | 12 joints
    # 每條腿：hip / thigh / calf

    # ================================================================
    # Simulation
    # ================================================================

    sim_dt: float = 0.005
    # Seconds | Isaac Lab physics timestep