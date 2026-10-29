# `cra373.urdf` 修改紀錄（vs. git HEAD）

這份文件記錄 `CRA373_model/cra373.urdf` 相對於 committed 版本（`git show HEAD:CRA373_model/cra373.urdf`）的每一項變更。
全部共五類：一個 joint velocity limit 的十倍錯誤、兩處 inertial 放錯位置、17 個 collision mesh 換成預先算好的 convex hull，以及檔案頂端新增的 `[sim]` 註解區塊。

前三項是物理上的錯誤修正，會直接改變模擬出來的動力學；第四項是載入時間的最佳化，PhysX 最後拿到的 collision shape 完全一樣。
在修正之前，模擬裡的關節可以用真機十倍的速度轉，兩條前腿有 2.9 kg 的質量掛在錯誤的位置，四隻腳的質量被放在膝蓋而不是腳尖。
這些都不是可以靠調 reward 蓋過去的誤差，所以在訓練之前要先修。

---

## 1. 摘要表

| 項目 | 舊值 | 新值 | 影響 |
|---|---|---|---|
| 12 個 revolute joint 的 `velocity` | `209.44` rad/s | `20.944` rad/s | PhysX 的 `velocity_limit_sim` 從 2000 rpm 修正為 200 rpm |
| `FR_hip` / `FL_hip` inertial `origin` 的 y | `+0.059012` | `-0.059012` | 每條前腿 1.454366 kg 的質心回到自己的 mesh 上（原本差 0.118 m） |
| `FR_hip` / `FL_hip` 的 `ixy`、`iyz` | 見 §3 | 兩項同時反號 | 慣量張量與鏡射後的幾何一致 |
| 四個 `*_foot` 的 inertial `origin` | `0 0 0` | 約 `(±0.0145, -0.2949, -0.1487)` | 0.05 kg 從膝蓋附近移到腳尖 |
| 四個 `*_foot` 的 inertia | `ixx=iyy=izz=1e-4` | `1.004e-05` / `5.041e-06` / `8.333e-06` | 移除膝關節上約 0.0055 kg·m² 的假慣量 |
| 17 個 `<collision>` 的 mesh | `meshes/<link>.stl` | `meshes/collision/<link>.stl` | 530,757 → 8,152 triangles，載入不必重算 hull |
| 檔案頂端 | 無 | `[sim]` 註解區塊 | 記錄上述變更與「刻意不改」清單 |

未變動：`<visual>` 的 17 個 mesh 參照、所有 link frame、所有 joint origin 與 `lower` / `upper` / `effort`、所有 mass 與 inertia 大小（總質量仍為 68.162843 kg）。

---

## 2. Joint velocity limit：209.44 → 20.944 rad/s

### 改了什麼

12 個 revolute joint（4 條腿 x hip / thigh / calf）的 `<limit ... velocity="209.44"/>` 全部改成 `velocity="20.944"`。
`lower`、`upper`、`effort="60.0"` 都沒有動。

### 為什麼

actuator model `cra373/actuator/XB42M.py` 的 `XB42MCfg.max_velocity` 寫的是 `math.radians(200.0 * 6.0)`，也就是模組輸出端空載轉速 200 rpm = 20.944 rad/s。
舊值 209.44 rad/s 換算是 2000 rpm，剛好是十倍，所以這是小數點的位移，不是減速比搞混：同一個檔案的 `gear_ratio` 是 9.0，不是 10。

同一份 config 的 T-N curve 也把 `tn_rpm_3` 定在 200 rpm，並且在該轉速把輸出扭矩降到 0，與 20.944 rad/s 一致。

### 影響

URDF 的 velocity limit 會被 Isaac Lab 寫進 PhysX 成為 `velocity_limit_sim`。
在 `D:\IsaacLab\source\isaaclab\isaaclab\actuators\actuator_base.py` 中，`velocity_limit_sim` 放在 `to_check` 清單裡，cfg 值為 `None` 時就採用 USD 裡的值，而 `XB42MCfg` 沒有覆寫它。

所以修正前，模擬裡的關節可以轉到真機十倍的速度。這對 RL 特別危險，因為 policy 會學會利用模擬裡才有的速度，sim-to-real 一定對不上。

---

## 3. 前腿 hip 的 inertial block 沿 y 鏡射（只有 `FR_hip` 與 `FL_hip`）

### 改了什麼

```
-      <origin xyz="0.020377 0.059012 0.000393" .../>   (FR_hip)
+      <origin xyz="0.020377 -0.059012 0.000393" .../>
-      <inertia ixx="0.002957" ixy="-0.000229" ixz="0.000003"
-               iyy="0.001744" iyz="-0.000001" izz="0.002087"/>
+      <inertia ixx="0.002957" ixy="0.000229" ixz="0.000003"
+               iyy="0.001744" iyz="0.000001" izz="0.002087"/>
```

`FL_hip` 同樣處理（它的 `ixy` 是 `+0.000229` → `-0.000229`，`iyz` 是 `-0.000001` → `+0.000001`）。
`ixx`、`iyy`、`izz`、`ixz` 與 `mass` 都沒有動。

### 為什麼

四個 hip link 的 mesh 是鏡射關係。實際量測 STL 的 bounding box 中心：

| mesh | 中心 x | 中心 y | 中心 z |
|---|---|---|---|
| `RR_hip` / `RL_hip` | ±0.01475 | **+0.06075** | 0 |
| `FR_hip` / `FL_hip` | ±0.01475 | **-0.06075** | 0 |

前腿的 inertial block 是從後腿整塊複製過來的，`origin` 的 y 留在 `+0.059012`，但幾何在 y = -0.0607。
對 x-z 平面做鏡射時，只有 `ixy` 與 `iyz` 這兩個乘積項會變號，`ixx` / `iyy` / `izz` / `ixz` 不變，所以只改了那兩項。

### 影響

每個 hip link 是 1.454366 kg，舊檔把它放在離自己幾何 0.118024 m 的地方，兩條前腿合計 2.9 kg 放錯位置。
對整機質心的影響是 `2 x 1.454366 x 0.118024 / 68.162843 = 0.00504 m`，也就是質心往後偏約 5 mm。

比質心偏移更麻煩的是，這讓前後腿的動力學不對稱，而且這個不對稱沒有任何物理來源。
Policy 會學到一組只在模擬裡成立的前後腿差異。

---

## 4. 腳掌的 inertial origin 與 inertia tensor（四個 `*_foot`）

### 改了什麼

舊值（四個腳掌都一樣）：

```
<inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.05"/>
  <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
```

新值（`origin` 為各自 mesh 的體積形心，逐腳略有差異）：

| link | origin x | origin y | origin z |
|---|---|---|---|
| `RR_foot` | -0.014499 | -0.294911 | -0.148672 |
| `FR_foot` | -0.014499 | -0.294911 | -0.148672 |
| `RL_foot` | +0.014501 | -0.294911 | -0.148671 |
| `FL_foot` | +0.014499 | -0.294912 | -0.148672 |

inertia 四個腳掌相同：`ixx="1.004e-05" iyy="5.041e-06" izz="8.333e-06"`，乘積項維持 0。`mass` 仍為 0.05 kg。

### 為什麼

`*_foot_fixed` 這個 fixed joint 的 origin 是 `(0, 0.154173, 0)`，也就是腳掌 link frame 落在 calf joint 再往前 0.154173 m 的位置。
但腳掌的 mesh 是 baked 進 STL 頂點的，實際 bounding box 中心在 `(±0.0145, -0.29527, -0.14751)`，距離 link frame 原點 0.3306 m。

所以舊檔的 `origin xyz="0 0 0"` 把腳掌質量放在膝蓋附近，而不是腳尖。

新的 inertia 是 20 x 40 x 28.5 mm 腳墊的實心長方體估計值（mesh bounding box 量到的尺寸剛好就是這三個數）：
`ixx = m/12·(b²+c²) = 1.005e-05`、`iyy = m/12·(a²+c²) = 5.05e-06`、`izz = m/12·(a²+b²) = 8.333e-06`，與檔案中的值一致。
舊的 `1e-4` 佔位值對應的等效半徑是 `sqrt(1e-4 / 0.05) = 0.045 m`，比整個腳掌還大。

### 影響

0.05 kg 放在 0.33 m 外，對膝關節的平行軸貢獻是 `m·r² = 0.05 x 0.109283 = 0.00546 kg·m²`。
calf link 自己的 `ixx` 只有 0.004290，所以這個假慣量比小腿本身還大，等於把膝關節的轉動慣量灌了一倍以上，而且方向是錯的。
這直接破壞擺腿動力學：swing phase 的加速度、落地時機、以及膝關節需要的扭矩全部算錯。

### 一個要記住的通則

腳掌 mesh 的 baked offset 就是為什麼**站立高度不能只靠 joint tree 算出來**。
`*_foot` 的 link origin 不是腳掌所在的位置，中間差了 0.33 m。
要知道實際離地高度，必須看 mesh 的頂點，或是用 `scripts/check_stand.py` 實際落地測。

---

## 5. Collision geometry 換成預先算好的 convex hull（17 個 `<collision>`）

### 改了什麼

舊檔每個 `<collision>` 都直接引用該 link 的完整 visual mesh，例如 `meshes/base_link.stl`（179,685 triangles）。
新檔改成 `meshes/collision/<link>.stl`，由新增的 `scripts/make_collision_meshes.py` 產生。

`logs/make-collision-meshes.txt` 的統計：

| link | src tris | hull tris | hull verts | src KB | out KB |
|---|---|---|---|---|---|
| `base_link` | 179,685 | 504 | 255 | 8,774 | 24.7 |
| `*_hip` (x4) | 36,196 | 506 | 255 | 1,767 | 24.8 |
| `*_thigh` (x4) | 42,592 | 506 | 255 | 2,080 | 24.8 |
| `*_calf` (x4) | 8,228 | 506 | 255 | 402 | 24.8 |
| `*_foot` (x4) | 752 | 392–396 | 198–200 | 37 | 19.2–19.4 |
| **TOTAL** | **530,757** | **8,152** | | | |

### 為什麼

`CRA373_model/config.yaml` 裡 Isaac Lab 的 URDF importer 設定是 `collider_type: convex_hull`（`collision_from_visuals: false`）。
也就是說 PhysX 本來就會把每個 collision mesh 縮成 convex hull，只是它每次把資產載入 physics scene 時都要從 53 萬個三角形重新 cook 一次。

自己先算好同一組 hull，PhysX 就直接拿到結果。
最後的 collision shape 完全沒變，因為 convex hull 的 convex hull 就是同一個 hull。這是載入時間的最佳化，不是物理變更。

### `<visual>` 沒有改

17 個 `<visual>` 仍然引用原始的全解析度 mesh，機器人看起來還是 CAD 匯出的樣子。
這一點要講明白，因為 patch script 的早期版本用一條 `<collision>.*?<mesh filename="` 的 regex 跨過了 link 邊界，連 `<visual>` 的參照一起改掉了。

現在的 `patch_urdf()` 先用 `re.subn(r"<collision>.*?</collision>", ...)` 切出區塊再改，並且額外掃一次 `<visual>` 把誤改的還原，是 idempotent 的，可以重複執行。
腳本最後會報告「`<visual>` blocks accidentally pointing at hulls」，預期值是 0。

---

## 6. 檔案頂端新增 `[sim]` 註解區塊

檔案開頭的 XML 註解裡加了一段 `[sim] 2026-09-15 modifications for Isaac Lab / RL training`，條列上面的第 1 到第 3 項，並附一份「NOT changed on purpose」清單。
在檔案內搜尋 `[sim]` 就能找到這一段。

> **注意**：這個區塊目前只列了第 1 到 3 項，**沒有**列第 5 項（collision hull）。
> 而且它的「NOT changed on purpose」裡還留著一行 `Collision geometry: every link still uses its visual mesh (convex_hull at import time).`，
> 這一行已經與檔案現況矛盾，是 collision 修改之前寫下、之後忘了更新的。下次動這個檔時要一併修掉。

---

## 7. 沒有改的東西

### Joint `<dynamics>`：仍然缺席

所有 joint 都沒有 `<dynamics>` 標籤，等於 `damping = 0`、`friction = 0`。
這是刻意的：關節摩擦已經在 `cra373/actuator/XB42M.py` 裡模型化了一次（`XB42MCfg.friction_nm = 0.67`，目前近似成 Coulomb friction 項）。
在 URDF 這邊再加一次會變成重複計算。

### `effort="60.0"`：不動

XB42M 是 explicit actuator。依 `actuator_base.py` 的 `_DEFAULT_MAX_EFFORT_SIM = 1.0e9`，當 `effort_limit_sim` 為 `None` 時 Isaac Lab 會把它設成 1e9，避免 solver 端再裁切一次。
真正的扭矩上限來自 `XB42MActuator.compute()` 裡的 T-N curve（0–75 rpm 峰值平台、75–175 rpm 下降、175–200 rpm 續降、200 rpm 以上輸出為 0）。
所以 URDF 的 `effort` 在模擬中不是有效上限，改它沒有意義。

### Mass 與 inertia 的量值：不動

只改了 inertia 的**位置**與腳掌那組明顯是佔位符的數值，所有 link 的 `mass` 一律未動。
機器人的擁有者已確認總質量 68.163 kg，檔案中 17 個 `mass` 相加為 68.162843 kg，`logs/build-usd.txt` 回報 68.163 kg。

### Link 座標系：不動

URDF 的前進方向是 -Y，不是常見的 +X。
這件事在環境程式碼裡處理（`cra373/envs/`），不靠改模型來「修正」，因為改 link frame 會讓所有 mesh 的 baked transform、joint origin 與既有的 USD 全部失效。

---

## 8. 改完之後要做什麼

**任何 URDF 修改都必須重新產生 USD。** 模擬讀的是 USD，不是 URDF；不重建的話上面所有修正都不會生效。

```bash
# Windows
set PYTHONPATH=D:/handson/torch27-shim
C:/isaac-sim/python.bat scripts/build_usd.py --headless

# Linux container
./isaaclab.sh -p scripts/build_usd.py --headless
```

輸出在 `CRA373_model/usd/`，這個目錄在 `.gitignore` 裡（`# generated by scripts/build_usd.py from CRA373_model/cra373.urdf`），不會進 repo。
`cra373/paths.py` 的 `usd_path()` 會優先使用 `CRA373_model/usd/cra373.usd`，存在就自動改用它，不必改任何設定。

`CRA373_model/configuration/` 底下的 USD 是 2026-08-21 由 UrdfConverter 產生的，早於本文所有修正，**不要**拿它來訓練。

USD 重建之後，訓練之前先跑落地測試：

```bash
C:/isaac-sim/python.bat scripts/check_stand.py --headless
```

它會檢查 spawn 高度是否正確、哪一組站姿的關節扭矩最省、以及 reset 時 actuator 有沒有打出扭矩尖峰。這三件事在腳掌 inertial 修正之後都會變，所以要重測。

如果動到 mesh，`scripts/make_collision_meshes.py` 也要重跑（它是 idempotent 的），再重建 USD。

---

## 9. 驗證

以下指令都在 repo 根目錄執行，每一條後面是它能證明的事。

```bash
# 完整 diff
git diff CRA373_model/cra373.urdf

# 12 個 joint 都改了，沒有漏掉任何一個，也沒有殘留舊值
grep -c 'velocity="20.944"' CRA373_model/cra373.urdf    # 12
grep -c 'velocity="209.44"' CRA373_model/cra373.urdf    # 0

# 總質量沒變
grep -o 'mass value="[0-9.]*"' CRA373_model/cra373.urdf \
  | grep -o '[0-9.]\+' | awk '{s+=$1} END {printf "%.6f (%d links)\n", s, NR}'
# -> 68.162843 (17 links)

# 17 個 collision 全部指向 hull，而且 17 個 visual 一個都沒被誤改
grep -c '<collision>' CRA373_model/cra373.urdf                 # 17
grep -c 'meshes/collision/' CRA373_model/cra373.urdf           # 17
grep -c '<visual>' CRA373_model/cra373.urdf                    # 17
grep -o 'filename="meshes/[A-Za-z_]*\.stl"' CRA373_model/cra373.urdf | wc -l   # 17（全解析度 mesh）

# 前腿 hip 的質心確實鏡射了
grep -A2 '<link name="FR_hip">' CRA373_model/cra373.urdf   # y = -0.059012
grep -A2 '<link name="RR_hip">' CRA373_model/cra373.urdf   # y = +0.059012
```

重建 USD 之後，`logs/build-usd.txt` 應該回報：

```
rigid bodies     : 17 (expected 17)
revolute joints  : 12 (expected 12)
total mass       : 68.163 kg (expected 68.163)
```

這三行同時成立，代表 URDF 的結構沒有在修改過程中被破壞（沒有多出或少掉 link、沒有把 revolute 改成 fixed、質量沒有被動到）。
`logs/` 在 `.gitignore` 裡，所以 `logs/build-usd.txt` 與 `logs/make-collision-meshes.txt` 都只存在於本機，不在 repo 裡。

### 已知的紀錄落差

`logs/make-collision-meshes.txt` 最後一行寫的是
`URDF <collision> mesh references pointing at meshes/collision/: 33 (expected 17)`。
那是 patch script 修好之前的舊版輸出（該版本的 regex 跨越了 link 邊界，把 visual 也算了進去），報告檔在腳本修正後沒有重新產生。
目前檔案的實測值是 collision 17、visual 誤改 0，以上面的 `grep` 為準。表格裡的三角形數字不受影響，因為 mesh 產生邏輯沒有變。
