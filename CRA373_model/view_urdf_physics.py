"""
CRA-373 URDF 物理模擬展示
===========================
有重力、有碰撞、有 PD 控制器

操作：
- 左鍵旋轉, 右鍵平移, 滾輪縮放
- tkinter 面板按鈕選擇動作
- q: 退出

需求：pip install mujoco numpy
"""

import os
import sys
import tempfile
import shutil
import time
import threading
import re
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("請先安裝：pip install mujoco")
    sys.exit(1)


# ╔══════════════════════════════════════════════════════════════╗
# ║                    場景建置                                  ║
# ╚══════════════════════════════════════════════════════════════╝

script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = None
for name in ["cra373.urdf", "cra373_full_new.urdf"]:
    p = os.path.join(script_dir, name)
    if os.path.exists(p):
        urdf_path = p
        break

if urdf_path is None:
    print("錯誤：找不到 cra373.urdf 或 cra373_full_new.urdf")
    sys.exit(1)

print(f"載入：{os.path.basename(urdf_path)}")

# 複製到英文臨時路徑（避免中文路徑問題）
tmp_dir = tempfile.mkdtemp(prefix="mujoco_cra373_")
tmp_urdf = os.path.join(tmp_dir, "robot.urdf")

with open(urdf_path, "r", encoding="utf-8") as f:
    urdf_string = f.read()
with open(tmp_urdf, "w", encoding="utf-8") as f:
    f.write(urdf_string)

urdf_dir = os.path.dirname(urdf_path)
for mesh_folder in ["meshes", "meshes_new"]:
    meshes_src = os.path.join(urdf_dir, mesh_folder)
    meshes_dst = os.path.join(tmp_dir, mesh_folder)
    if os.path.exists(meshes_src):
        shutil.copytree(meshes_src, meshes_dst)

# URDF → MJCF → 完整物理場景
tmp_model = mujoco.MjModel.from_xml_path(tmp_urdf)
mjcf_path = os.path.join(tmp_dir, "robot.xml")
mujoco.mj_saveLastXML(mjcf_path, tmp_model)

with open(mjcf_path, "r", encoding="utf-8") as f:
    mjcf_content = f.read()

# 加入地面和光源
ground_xml = """
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"
          contype="1" conaffinity="1" condim="3" friction="1.5 0.01 0.001"/>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
    <light pos="2 2 2" dir="-1 -1 -1" diffuse="0.5 0.5 0.5"/>
"""
mjcf_content = mjcf_content.replace("<worldbody>", "<worldbody>" + ground_xml, 1)

# 去掉 actuatorfrcrange 限制
mjcf_content = re.sub(r' actuatorfrcrange="[^"]*"', '', mjcf_content)

# 確保 geom 有碰撞
mjcf_content = mjcf_content.replace('contype="0"', 'contype="1"')
mjcf_content = mjcf_content.replace('conaffinity="0"', 'conaffinity="1"')

# 把整隻狗包在 freejoint body 裡
light_pattern = r'(<light[^/]*/>\s*\n)'
lights = list(re.finditer(light_pattern, mjcf_content))
if lights:
    last_light_end = lights[-1].end()
    body_open = '    <body name="robot_base" pos="0 0 0.25">\n      <freejoint name="root"/>\n'
    mjcf_content = mjcf_content[:last_light_end] + body_open + mjcf_content[last_light_end:]
    mjcf_content = mjcf_content.replace("  </worldbody>", "    </body>\n  </worldbody>")

# 加 PD actuator
joint_names_in_xml = re.findall(r'<joint\s+name="([^"]+)"', mjcf_content)
joint_names_in_xml = [n for n in joint_names_in_xml if n != 'root']

actuator_xml = "\n  <actuator>\n"
for jname in joint_names_in_xml:
    actuator_xml += f'    <position name="act_{jname}" joint="{jname}" kp="800" dampratio="1"/>\n'
actuator_xml += "  </actuator>\n"
mjcf_content = mjcf_content.replace("</mujoco>", actuator_xml + "</mujoco>")

# 加 option 和 default
option_default = """
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <joint damping="15" armature="0.05"/>
    <geom condim="3" contype="1" conaffinity="1" friction="1.5 0.01 0.001"/>
  </default>
"""
mjcf_content = re.sub(r'(<mujoco[^>]*>)', r'\1' + option_default, mjcf_content, count=1)

# 寫入場景檔
scene_path = os.path.join(tmp_dir, "scene_physics.xml")
with open(scene_path, "w", encoding="utf-8") as f:
    f.write(mjcf_content)

print(f"[場景檔] {scene_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                    模型載入與初始化                           ║
# ╚══════════════════════════════════════════════════════════════╝

model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)

print(f"✓ 物理場景載入成功")
print(f"  joints: {model.njnt}, actuators: {model.nu}, bodies: {model.nbody}")

# 收集可控關節
hinge_joints = []
for i in range(model.njnt):
    if model.jnt_type[i] == 3:  # hinge
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
        qpos_idx = model.jnt_qposadr[i]
        lo, hi = model.jnt_range[i] if model.jnt_limited[i] else (-3.14, 3.14)
        hinge_joints.append({
            'id': i, 'name': name, 'qpos_idx': qpos_idx,
            'lower': float(lo), 'upper': float(hi)
        })

joint_map = {j['name']: idx for idx, j in enumerate(hinge_joints)}

# joint name → actuator id
act_map = {}
for act_id in range(model.nu):
    act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
    if act_name:
        jname = act_name.replace("act_", "", 1)
        act_map[jname] = act_id

print(f"\n可控關節 ({len(hinge_joints)} 個)：")
for idx, j in enumerate(hinge_joints):
    has_act = "✓" if j['name'] in act_map else "✗"
    print(f"  {idx}: {j['name']}  {has_act}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                    控制工具函式                               ║
# ╚══════════════════════════════════════════════════════════════╝

def set_target(name, angle_deg):
    """設定單一關節目標角度 (度)"""
    if name in joint_map and name in act_map:
        j = hinge_joints[joint_map[name]]
        rad = np.clip(np.radians(angle_deg), j['lower'], j['upper'])
        data.ctrl[act_map[name]] = rad


def set_all_targets(target_dict):
    """設定多個關節目標"""
    for name, angle_deg in target_dict.items():
        set_target(name, angle_deg)


def sim_steps(n=10):
    """執行 n 步物理模擬"""
    for _ in range(n):
        mujoco.mj_step(model, data)


# 穩定站立
for j in hinge_joints:
    set_target(j['name'], 0)

print("\n穩定中...")
for _ in range(3000):
    mujoco.mj_step(model, data)
print("✓ 穩定完成")


def smooth_to(target_angles, duration=1.5, fps=30):
    """平滑過渡到目標角度（ease in-out）"""
    start_angles = {}
    for name in target_angles:
        if name in act_map:
            start_angles[name] = np.degrees(data.ctrl[act_map[name]])
        elif name in joint_map:
            j = hinge_joints[joint_map[name]]
            start_angles[name] = np.degrees(data.qpos[j['qpos_idx']])

    steps = int(duration * fps)
    for step in range(steps + 1):
        if not viewer.is_running():
            return
        t = step / steps
        t = t * t * (3 - 2 * t)  # smoothstep
        for name, target in target_angles.items():
            if name in start_angles:
                current = start_angles[name] + (target - start_angles[name]) * t
                set_target(name, current)
        sim_steps(int(1.0 / (fps * model.opt.timestep)))
        viewer.sync()
        time.sleep(1.0 / fps)


# ╔══════════════════════════════════════════════════════════════╗
# ║                    關節方向約定                               ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  hip_joint   axis=Y  limit [-90°, 90°]   +外展  -內收       ║
# ║  thigh_joint axis=X  limit [-90°, 45°]   -前收(蹲) +後伸    ║
# ║  calf_joint  axis=X  limit [-62°, 62°]   +伸直  -屈膝(蹲)   ║
# ║  0° = 站立姿態                                              ║
# ╚══════════════════════════════════════════════════════════════╝

STAND_HIP = 0
STAND_THIGH = 0
STAND_CALF = 0


# ╔══════════════════════════════════════════════════════════════╗
# ║                    姿態定義                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def pose_stand():
    """站立：所有關節歸零"""
    target = {}
    for j in hinge_joints:
        target[j['name']] = 0
    return target


def pose_sit():
    """坐下（Go2 風格）：後腿完全折疊，前腿直立"""
    target = pose_stand()
    target['RR_thigh_joint'] = -88
    target['RL_thigh_joint'] = -88
    target['RR_calf_joint'] = -62
    target['RL_calf_joint'] = -62
    target['FR_thigh_joint'] = 10
    target['FL_thigh_joint'] = 10
    target['FR_calf_joint'] = 45
    target['FL_calf_joint'] = 45
    return target


def pose_beg():
    """作揖：前腿高抬蜷縮，後腿深蹲支撐"""
    target = pose_stand()
    target['FR_thigh_joint'] = -50
    target['FL_thigh_joint'] = -50
    target['FR_calf_joint'] = -37
    target['FL_calf_joint'] = -37
    target['RR_thigh_joint'] = -25
    target['RL_thigh_joint'] = -25
    target['RR_calf_joint'] = -30
    target['RL_calf_joint'] = -30
    return target


def pose_stretch():
    """伸展 (play bow)：前腿撐高，後腿趴低"""
    target = pose_stand()
    target['FR_thigh_joint'] = 20
    target['FL_thigh_joint'] = 20
    target['FR_calf_joint'] = -20
    target['FL_calf_joint'] = -20
    target['RR_thigh_joint'] = -30
    target['RL_thigh_joint'] = -30
    target['RR_calf_joint'] = 25
    target['RL_calf_joint'] = 25
    return target


# ╔══════════════════════════════════════════════════════════════╗
# ║                    動畫定義                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def anim_sit(fps=30):
    """坐下（Go2 風格）：兩階段平滑折疊"""
    phase1 = pose_stand()
    phase1['RR_thigh_joint'] = -35
    phase1['RL_thigh_joint'] = -35
    phase1['RR_calf_joint'] = -30
    phase1['RL_calf_joint'] = -30
    phase1['FR_thigh_joint'] = 5
    phase1['FL_thigh_joint'] = 5
    smooth_to(phase1, duration=1.0)
    smooth_to(pose_sit(), duration=1.2)


def anim_walk(cycles=10, cycle_time=0.8, fps=30):
    """走路（對角步態）"""
    steps_per_frame = int(1.0 / (fps * model.opt.timestep))
    steps_per_cycle = int(cycle_time * fps)

    swing_ratio = 0.35
    thigh_range = 15
    lift_height = 18
    crouch = 5
    crouch_calf = 5

    thigh_center = STAND_THIGH - crouch
    calf_stance = STAND_CALF - crouch_calf

    def leg_phase(t):
        if t < swing_ratio:
            p = t / swing_ratio
            sp = 0.5 - 0.5 * np.cos(p * np.pi)
            thigh = thigh_center + thigh_range - 2 * thigh_range * sp
            calf = calf_stance - lift_height * np.sin(p * np.pi) ** 0.7
            return thigh, calf
        else:
            p = (t - swing_ratio) / (1.0 - swing_ratio)
            sp = 0.5 - 0.5 * np.cos(p * np.pi)
            thigh = thigh_center - thigh_range + 2 * thigh_range * sp
            return thigh, calf_stance

    for cycle in range(cycles):
        for step in range(steps_per_cycle):
            if not viewer.is_running():
                return
            t = step / steps_per_cycle
            t_a = t % 1.0
            t_b = (t + 0.5) % 1.0

            th_a, ca_a = leg_phase(t_a)
            th_b, ca_b = leg_phase(t_b)

            set_target('FR_thigh_joint', th_a)
            set_target('FR_calf_joint', ca_a)
            set_target('RL_thigh_joint', th_a)
            set_target('RL_calf_joint', ca_a)
            set_target('RR_thigh_joint', th_b)
            set_target('RR_calf_joint', ca_b)
            set_target('FL_thigh_joint', th_b)
            set_target('FL_calf_joint', ca_b)

            sim_steps(steps_per_frame)
            viewer.sync()
            time.sleep(1.0 / fps)


def anim_trot(cycles=8, cycle_time=0.5, fps=30):
    """小跑（對角步態，節奏較快）"""
    steps_per_frame = int(1.0 / (fps * model.opt.timestep))
    steps_per_cycle = int(cycle_time * fps)

    swing_ratio = 0.35
    thigh_range = 15
    lift_height = 18
    crouch = 5

    thigh_center = STAND_THIGH - crouch
    calf_stance = STAND_CALF - crouch

    def leg_phase(t):
        if t < swing_ratio:
            p = t / swing_ratio
            sp = 0.5 - 0.5 * np.cos(p * np.pi)
            thigh = thigh_center + thigh_range - 2 * thigh_range * sp
            calf = calf_stance - lift_height * np.sin(p * np.pi)
            return thigh, calf
        else:
            p = (t - swing_ratio) / (1.0 - swing_ratio)
            sp = 0.5 - 0.5 * np.cos(p * np.pi)
            thigh = thigh_center - thigh_range + 2 * thigh_range * sp
            return thigh, calf_stance

    for cycle in range(cycles):
        for step in range(steps_per_cycle):
            if not viewer.is_running():
                return
            t = step / steps_per_cycle
            t_a = t % 1.0
            t_b = (t + 0.5) % 1.0

            th_a, ca_a = leg_phase(t_a)
            th_b, ca_b = leg_phase(t_b)

            set_target('FR_thigh_joint', th_a)
            set_target('FR_calf_joint', ca_a)
            set_target('RL_thigh_joint', th_a)
            set_target('RL_calf_joint', ca_a)
            set_target('RR_thigh_joint', th_b)
            set_target('RR_calf_joint', ca_b)
            set_target('FL_thigh_joint', th_b)
            set_target('FL_calf_joint', ca_b)

            sim_steps(steps_per_frame)
            viewer.sync()
            time.sleep(1.0 / fps)


def anim_walk_back(cycles=6, cycle_time=1.0, fps=30):
    """倒退走（對角步態，反向）"""
    steps_per_frame = int(1.0 / (fps * model.opt.timestep))
    steps_per_cycle = int(cycle_time * fps)

    swing_ratio = 0.35
    thigh_range = 15
    lift_height = 18
    crouch = 8
    crouch_calf = 10

    thigh_center = STAND_THIGH - crouch
    calf_stance = STAND_CALF - crouch_calf

    def leg_phase(t):
        if t < swing_ratio:
            p = t / swing_ratio
            sp = 0.5 - 0.5 * np.cos(p * np.pi)
            thigh = thigh_center - thigh_range + 2 * thigh_range * sp
            calf = calf_stance - lift_height * np.sin(p * np.pi)
            return thigh, calf
        else:
            p = (t - swing_ratio) / (1.0 - swing_ratio)
            sp = 0.5 - 0.5 * np.cos(p * np.pi)
            thigh = thigh_center + thigh_range - 2 * thigh_range * sp
            return thigh, calf_stance

    for cycle in range(cycles):
        for step in range(steps_per_cycle):
            if not viewer.is_running():
                return
            t = step / steps_per_cycle
            t_a = t % 1.0
            t_b = (t + 0.5) % 1.0

            th_a, ca_a = leg_phase(t_a)
            th_b, ca_b = leg_phase(t_b)

            set_target('FR_thigh_joint', th_a)
            set_target('FR_calf_joint', ca_a)
            set_target('RL_thigh_joint', th_a)
            set_target('RL_calf_joint', ca_a)
            set_target('RR_thigh_joint', th_b)
            set_target('RR_calf_joint', ca_b)
            set_target('FL_thigh_joint', th_b)
            set_target('FL_calf_joint', ca_b)

            sim_steps(steps_per_frame)
            viewer.sync()
            time.sleep(1.0 / fps)


def anim_wave(cycles=5, cycle_time=0.6, fps=30):
    """揮手：重心轉移 → 抬右前腿 → hip 左右揮動"""
    smooth_to(pose_stand(), duration=0.4)

    # 重心轉移到左側三足
    shift = pose_stand()
    shift['FL_hip_joint'] = 15
    shift['RR_hip_joint'] = 15
    shift['RL_hip_joint'] = 15
    shift['FL_thigh_joint'] = -5
    shift['FL_calf_joint'] = -5
    smooth_to(shift, duration=0.8)

    # 抬起右前腿
    lift = dict(shift)
    lift['FR_thigh_joint'] = -50
    lift['FR_calf_joint'] = -37
    smooth_to(lift, duration=0.5)

    # 左右揮動
    steps_per_frame = int(1.0 / (fps * model.opt.timestep))
    steps_per_cycle = int(cycle_time * fps)
    for cycle in range(cycles):
        for step in range(steps_per_cycle):
            if not viewer.is_running():
                return
            t = step / steps_per_cycle
            phase = t * 2 * np.pi
            set_target('FR_hip_joint', np.sin(phase) * 35)
            set_target('FR_calf_joint', -37 + np.sin(phase * 2) * 5)
            sim_steps(steps_per_frame)
            viewer.sync()
            time.sleep(1.0 / fps)

    # 收腿回站姿
    smooth_to(shift, duration=0.6)
    smooth_to(pose_stand(), duration=0.4)


def anim_bounce(cycles=5, cycle_time=0.6, fps=30):
    """原地彈跳：蹲下回彈"""
    steps_per_frame = int(1.0 / (fps * model.opt.timestep))
    steps_per_cycle = int(cycle_time * fps)
    for cycle in range(cycles):
        for step in range(steps_per_cycle):
            if not viewer.is_running():
                return
            t = step / steps_per_cycle
            bend = max(0, np.sin(t * 2 * np.pi)) * 25

            for j in hinge_joints:
                if 'thigh' in j['name']:
                    set_target(j['name'], STAND_THIGH - bend)
                elif 'calf' in j['name']:
                    set_target(j['name'], STAND_CALF - bend)

            sim_steps(steps_per_frame)
            viewer.sync()
            time.sleep(1.0 / fps)


def anim_look_around(fps=30):
    """左右看：後腿為支點，前腿差動帶動身體 yaw"""
    steps_per_frame = int(1.0 / (fps * model.opt.timestep))

    front_diff = 18
    rear_anchor = 5
    calf_comp = 8
    hip_shift = 3
    scan_time = 1.0
    hold_time = 0.3
    cycles = 2

    # 後腿微蹲壓低重心
    prep = pose_stand()
    prep['RR_thigh_joint'] = -rear_anchor
    prep['RL_thigh_joint'] = -rear_anchor
    prep['RR_calf_joint'] = -rear_anchor
    prep['RL_calf_joint'] = -rear_anchor
    smooth_to(prep, 0.5)

    for _ in range(cycles):
        # 向左看
        look_left = pose_stand()
        look_left['FR_thigh_joint'] = front_diff
        look_left['FR_calf_joint'] = calf_comp
        look_left['FL_thigh_joint'] = -front_diff
        look_left['FL_calf_joint'] = -calf_comp
        look_left['RR_thigh_joint'] = rear_anchor
        look_left['RR_calf_joint'] = calf_comp
        look_left['RL_thigh_joint'] = -rear_anchor
        look_left['RL_calf_joint'] = -rear_anchor
        look_left['FR_hip_joint'] = hip_shift
        look_left['FL_hip_joint'] = hip_shift
        smooth_to(look_left, scan_time)

        for _ in range(int(hold_time * fps)):
            if not viewer.is_running():
                return
            sim_steps(steps_per_frame)
            viewer.sync()
            time.sleep(1.0 / fps)

        # 向右看
        look_right = pose_stand()
        look_right['FR_thigh_joint'] = -front_diff
        look_right['FR_calf_joint'] = -calf_comp
        look_right['FL_thigh_joint'] = front_diff
        look_right['FL_calf_joint'] = calf_comp
        look_right['RR_thigh_joint'] = -rear_anchor
        look_right['RR_calf_joint'] = -rear_anchor
        look_right['RL_thigh_joint'] = rear_anchor
        look_right['RL_calf_joint'] = calf_comp
        look_right['FR_hip_joint'] = -hip_shift
        look_right['FL_hip_joint'] = -hip_shift
        smooth_to(look_right, scan_time * 2)

        for _ in range(int(hold_time * fps)):
            if not viewer.is_running():
                return
            sim_steps(steps_per_frame)
            viewer.sync()
            time.sleep(1.0 / fps)

    smooth_to(pose_stand(), 0.6)


# ╔══════════════════════════════════════════════════════════════╗
# ║                    動作序列 & GUI                            ║
# ╚══════════════════════════════════════════════════════════════╝

demos = [
    ("站立",     lambda: smooth_to(pose_stand(), 1.0)),
    ("左右看",   anim_look_around),
    ("走路",     lambda: anim_walk(cycles=6)),
    ("倒退走",   lambda: anim_walk_back(cycles=4)),
    ("小跑",     lambda: anim_trot(cycles=6)),
    ("原地彈跳", lambda: anim_bounce(cycles=5)),
    ("坐下",     anim_sit),
    ("作揖",     lambda: smooth_to(pose_beg(), 1.5)),
    ("揮手",     lambda: anim_wave(cycles=5)),
    ("伸展",     lambda: smooth_to(pose_stretch(), 1.5)),
    ("歸零",     lambda: smooth_to(pose_stand(), 1.0)),
]

# 啟動 viewer
viewer = mujoco.viewer.launch_passive(model, data)

# 動作執行旗標
action_running = [False]


def run_action(idx):
    """在背景執行動作"""
    if action_running[0]:
        return

    def _run():
        action_running[0] = True
        name, action = demos[idx]
        print(f"\n▶ {name}")
        try:
            action()
        finally:
            action_running[0] = False

    threading.Thread(target=_run, daemon=True).start()


# tkinter 按鈕面板
try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TK = True
except ImportError:
    HAS_TK = False

if HAS_TK:
    def create_button_gui():
        root = tk.Tk()
        root.title("CRA-373 動作控制")
        root.geometry("280x420")
        root.resizable(True, True)
        root.attributes('-topmost', True)

        ttk.Label(root, text="CRA-373 動作控制面板", font=("", 12, "bold")).pack(pady=10)

        for idx, (name, _) in enumerate(demos):
            ttk.Button(
                root, text=name, width=20,
                command=lambda i=idx: run_action(i)
            ).pack(pady=3, padx=20)

        ttk.Separator(root, orient='horizontal').pack(fill='x', padx=10, pady=10)
        ttk.Button(
            root, text="退出", width=20,
            command=lambda: (viewer.close(), root.destroy())
        ).pack(pady=5)

        def check_viewer():
            if not viewer.is_running():
                root.destroy()
                return
            root.after(500, check_viewer)

        root.after(500, check_viewer)
        root.protocol("WM_DELETE_WINDOW", lambda: (viewer.close(), root.destroy()))
        root.mainloop()

    threading.Thread(target=create_button_gui, daemon=True).start()
else:
    def input_loop():
        print("\n" + "=" * 50)
        print("  CRA-373 物理模擬展示")
        print("  輸入編號選擇動作 | q: 退出")
        print("=" * 50)
        for idx, (name, _) in enumerate(demos):
            print(f"  {idx}: {name}")
        print()
        while viewer.is_running():
            try:
                cmd = input("> ").strip()
                if cmd.lower() == 'q':
                    viewer.close()
                    break
                if cmd.isdigit() and 0 <= int(cmd) < len(demos):
                    run_action(int(cmd))
            except (EOFError, KeyboardInterrupt):
                break

    threading.Thread(target=input_loop, daemon=True).start()


# ╔══════════════════════════════════════════════════════════════╗
# ║                    主迴圈                                    ║
# ╚══════════════════════════════════════════════════════════════╝

time.sleep(0.5)

while viewer.is_running():
    if not action_running[0]:
        sim_steps(int(1.0 / (30 * model.opt.timestep)))
        viewer.sync()
    time.sleep(1.0 / 30.0)

shutil.rmtree(tmp_dir, ignore_errors=True)
print("\n已關閉。")
