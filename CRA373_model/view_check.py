"""
CRA-373 Isaac Sim 版 URDF 快速驗證
====================================
用 MuJoCo 載入確認：
1. mesh 是否正確顯示（無側躺）
2. 關節是否正常運作
3. 所有腿是否對稱

操作：左鍵旋轉, 右鍵平移, 滾輪縮放
需求：pip install mujoco numpy
"""

import os
import sys
import tempfile
import shutil
import time
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("請先安裝：pip install mujoco")
    sys.exit(1)

# ===== 載入 URDF =====
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, "cra373.urdf")

if not os.path.exists(urdf_path):
    print(f"錯誤：找不到 {urdf_path}")
    sys.exit(1)

# 複製到英文暫存路徑 (避免中文路徑問題)
tmp_dir = tempfile.mkdtemp(prefix="mujoco_check_")
tmp_urdf = os.path.join(tmp_dir, "cra373.urdf")
shutil.copy2(urdf_path, tmp_urdf)
shutil.copytree(os.path.join(script_dir, "meshes"), os.path.join(tmp_dir, "meshes"))

print(f"載入: cra373.urdf (Isaac Sim 版)")
print(f"暫存: {tmp_dir}")

# ===== 載入模型 =====
model = mujoco.MjModel.from_xml_path(tmp_urdf)
data = mujoco.MjData(model)

# 無重力模式，方便觀察姿態
model.opt.gravity[:] = [0, 0, 0]

# 抬高 base 方便觀察
if model.njnt > 0 and model.jnt_type[0] == 0:
    data.qpos[2] = 0.35

# ===== 收集關節資訊 =====
hinge_joints = []
for i in range(model.njnt):
    if model.jnt_type[i] == 3:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
        qpos_idx = model.jnt_qposadr[i]
        lo, hi = model.jnt_range[i] if model.jnt_limited[i] else (-3.14, 3.14)
        hinge_joints.append({
            'id': i, 'name': name, 'qpos_idx': qpos_idx,
            'lower': float(lo), 'upper': float(hi)
        })

print(f"\n✓ 載入成功")
print(f"  bodies: {model.nbody}, joints: {model.njnt}, geoms: {model.ngeom}")
print(f"\n可控關節 ({len(hinge_joints)} 個)：")
for j in hinge_joints:
    print(f"  {j['name']:20s}  [{np.degrees(j['lower']):7.1f}° ~ {np.degrees(j['upper']):6.1f}°]")

# ===== 設定站立姿態 =====
print("\n設定站立姿態: hip=0°, thigh=-35°, calf=25°")
for j in hinge_joints:
    if 'hip' in j['name']:
        data.qpos[j['qpos_idx']] = 0
    elif 'thigh' in j['name']:
        data.qpos[j['qpos_idx']] = np.radians(-35)
    elif 'calf' in j['name']:
        data.qpos[j['qpos_idx']] = np.radians(25)

mujoco.mj_forward(model, data)

# ===== 驗證項目 =====
print("\n=== 驗證檢查 ===")

# 檢查 base_link 方向 (Z-up: 機身應水平)
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
if base_id >= 0:
    print(f"  ✓ base_link 位置: {data.xpos[base_id]}")

# 檢查四腿腳掌高度是否對稱
foot_names = ['RR_foot', 'FR_foot', 'RL_foot', 'FL_foot']
foot_z = []
for fn in foot_names:
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fn)
    if fid >= 0:
        z = data.xpos[fid][2]
        foot_z.append(z)
        print(f"  {fn} Z={z:.4f} m")

if foot_z:
    spread = max(foot_z) - min(foot_z)
    if spread < 0.01:
        print(f"  ✓ 四腳高度一致 (差異 {spread*1000:.1f} mm)")
    else:
        print(f"  ⚠ 四腳高度不一致! 差異 {spread*1000:.1f} mm")

print("\n開啟 MuJoCo viewer... (關閉視窗退出)")

# ===== 開啟 viewer =====
viewer = mujoco.viewer.launch_passive(model, data)

# 簡單動畫: 緩慢擺動所有關節確認運動方向
print("自動擺動關節中 (確認運動方向)...")
t = 0
while viewer.is_running():
    t += 1.0 / 60.0
    
    # 每 3 秒一個週期，緩慢擺動
    phase = np.sin(t * 2 * np.pi / 3.0) * 0.3  # ±0.3 rad ≈ ±17°
    
    for j in hinge_joints:
        if 'thigh' in j['name']:
            # thigh: -35° ± 17°
            data.qpos[j['qpos_idx']] = np.clip(
                np.radians(-35) + phase, j['lower'], j['upper'])
        elif 'calf' in j['name']:
            # calf: 25° ± 10° (反向)
            data.qpos[j['qpos_idx']] = np.clip(
                np.radians(25) - phase * 0.6, j['lower'], j['upper'])
    
    mujoco.mj_forward(model, data)
    viewer.sync()
    time.sleep(1.0 / 60.0)

shutil.rmtree(tmp_dir, ignore_errors=True)
print("\n已關閉。")
