# Stairs and gap showcase validation

The implementation and all validation runs are inside `CRA373_Affordance_LAS`.

| Run | Command options | Observed result |
| --- | --- | --- |
| Headless PPO cycle | `--headless --frames 70 --showcase-cycle-seconds 0.4 --dashboard-port 0 --no-browser` | Existing flat PPO actor executed and switched stairs → gap → rough → stairs; gap had 54 unknown cells. |
| GUI PPO cycle | `--frames 85 --showcase-cycle-seconds 0.4 --no-browser --dashboard-port 0` | Kit viewport updated 187 stair markers, 133 gap support markers, and 187 rough markers while switching all three tiles. |
| All-rough regression | `--terrain-scene rough --headless --frames 20 --control neutral --dashboard-port 0 --no-browser` | 187/187 valid rough-terrain hits. |

`showcase_stairs_viewport.png` shows the stair marker layout. `showcase_gap_viewport_final.png` shows the revised overview with stairs and the actual square gap. `showcase_gui_cycle.log`, `showcase_flat_ppo_cycle.log`, and `rough_mode_regression.log` contain the corresponding runtime traces.

Gray cells over the gap are missing RayCaster hits; they are not labelled as inferred physical gaps. The hole itself has no world-surface position, so it has no 3D marker. These runs demonstrate presentation and computation, not validated PPO gap crossing or learned LAS inference.
