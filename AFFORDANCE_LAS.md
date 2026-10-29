# CRA373 terrain affordance demo

## Repository analysis and copy verification

This project was copied in full from `../CRA373_final` before development. The initial `diff -qr` comparison was empty. The copied tree includes 196 PPO checkpoint files, the robot USD/URDF and meshes, training scripts, and existing configuration. Baseline checks in the copy passed: Python compilation, URDF parsing (17 links, 16 joints), and imports of the existing offline placeholder modules. The sampled `model_4200.pt` loaded successfully with Torch’s restricted loader and contains actor, critic and optimizer state dictionaries. All new work and generated results are in this folder.

| Existing component | Finding / reuse |
| --- | --- |
| Robot | `CRA373_model/cra373.urdf`, `cra373.usd`, mesh assets; `cra373/envs/CRA373.py` supplies articulation and actuator config. Twelve actuated hip/thigh/calf joints; the URDF contains joint limits. No validated step, jump, body clearance, leg reach or stability limits were found. |
| Environments | Direct RL flat and rough configs in `cra373/envs/cra373_env_cfg.py`; manager-based flat, rough, jump and parkour configs in `cra373/envs/*_env_cfg.py`. |
| Actions / observations | Direct environment action space is 12 joint-position offsets with scale 0.5. Flat policy observation is 48 values. Rough policy observation is 235: 48 proprioceptive/command/action values plus 187 clipped height samples. These clipped policy inputs are unsuitable for metric geometry, so the new adapter reads raw ray hits. |
| Sensors | Direct rough environment uses a yaw-aligned RayCaster height grid, 1.6 × 1.0 m at 0.1 m resolution. A camera config and acquisition code are commented out. RGB and depth capture are therefore unavailable. A point cloud marker utility exists in the direct environment. |
| Terrain | Isaac Lab rough terrain generator and local parkour terrain configs for gaps, platforms, boxes, stairs and stones. |
| Training / playback | `scripts/rsl_rl/train.py`, `play.py`, PPO configs and 196 flat-run checkpoints in `logs/rsl_rl/cra373_flat`. `play_affordance.py`, prior perception, affordance and skill selector files are empty placeholders. No rollout dataset or trained LAS was found. |
| Installed environment | Base Python 3.14.6 has NumPy and Matplotlib but no Torch/Isaac Lab. `env_isaaclab_2` has Torch 2.10.0+cu128, Isaac Lab package metadata 6.1.17, Isaac Sim 6.0.0.1 and an importable RSL-RL module. Isaac Sim GUI and headless runs were launched and verified in this copy. |

## Implemented architecture

`TerrainObservation` → raw height grid → `estimate_affordance` → spatial masks and metric features → `reason` → candidate action scores and explanation → browser or saved results.

The grid has 17 rows (+x forward) and 11 columns (+y left), at 0.1 m/cell. Height is world-up elevation in metres. `from_isaac_ray_hits` takes world-frame RayCaster hit XYZ, base world position and yaw; it bins into a base-yaw-aligned grid without assuming ray order. Roll and pitch do not tilt the map's vertical axis. Missing hits become **unknown**. A **known gap** needs separate support truth from a synthetic scenario or another validated sensor process. This prevents a ray miss from being presented as a measured physical gap.

`CapabilityHypotheses` contains explicit demo thresholds for step height, slope, roughness, discontinuity and landing length. They are **unvalidated hypotheses**, not CRA373 physical limits. Traversable and foothold masks are geometric estimates only. The reasoning output also includes 12 URDF joint-limit pairs and explicitly unknown body dimensions, leg reach and tilt capability. Height scans contain no appearance, friction or material information. The action scores are **uncalibrated heuristics**, not probabilities. Step Over / Climb can be the research recommendation while still marked unsupported for execution. Jump has no feasibility score or validated controller. Avoid has no connected route planner. No recommended action is sent to the robot.

## Live Isaac Sim and dashboard

To open the terrain in a visible Isaac Sim Kit window and the live affordance dashboard in a separate browser window, run from this copied project:

```bash
/home/dog101/miniconda3/envs/env_isaaclab_2/bin/python scripts/rsl_rl/live_affordance.py
```

The dashboard URL is `http://127.0.0.1:7860/?mode=live`. The launcher starts a single `CRA373-Rough-v0` environment with only seeded `random_rough` height-field tiles, reads raw `height_scanner.data.ray_hits_w` and robot pose at the same simulation step, rotates hits using Isaac Lab's **xyzw** quaternion convention, and publishes computed affordances and reasoning. The browser polls every 500 ms. It can also show the four offline scenarios from its selector.

This live mode defaults to read-only playback of the existing flat PPO actor (`model_4200.pt`) on rough terrain. The rough observation contains the same 36 proprioception/command values, then 187 height samples, then the same 12 previous-action values; the adapter selects the matching 48 values for the flat actor. This demonstrates policy execution but is an **unvalidated flat-to-rough transfer**, not a trained rough locomotion policy. Use `--control neutral` for zero joint offsets. Affordance recommendations do not enter the action stream. Use `--no-browser` to suppress automatic browser opening, `--dashboard-port` to change the port, and `--frames N` for a finite run. Closing the Isaac Sim window stops the dashboard server.

## Launch

From this copied project directory:

```bash
python -m cra373.demo.demo_offline --output demo_results
python -m cra373.demo.demo_app --port 7860
```

Open `http://127.0.0.1:7860`. Offline mode needs NumPy and Matplotlib. The browser app uses only the Python standard library and NumPy. It does not launch Isaac Sim or require a GUI renderer. The offline dashboard shows observation, spatial masks, reasoning, action feasibility and the honest LAS-unavailable state. Live mode adds a matching heuristic traversability color grid.

The four fixed synthetic scenarios are Flat, Rough, Raised step and Known gap. `demo_results/*.json` contains computed input maps, masks, features, actions and per-module latency. Four PNG figures and `summary.csv` are generated from those actual computations. The synthetic gap includes explicit support truth; no sensor or real-world accuracy is inferred.

For another Isaac Lab collector, call `from_isaac_ray_hits(ray_hits_w_m, base_position_w_m, base_yaw_rad)` at the same simulation step as the policy observation and action. Use the raw `height_scanner.data.ray_hits_w` tensor converted to NumPy for one environment. Record the simulation timestamp and base velocity with that frame. The adapter is importable without Isaac; an end-to-end Isaac rollout collector is still future work.

## Evaluation and reproducibility

Run focused checks:

```bash
python -m unittest discover -s tests -v
```

The CSV records measured Python geometry and reasoning wall time on one frame at a time, including total. It excludes simulation, ray casting, browser rendering and file output. The scenario seed is fixed at 373. The action recommendations are regression checks against synthetic cases; they are **not** accuracy on validated capability labels. mIoU, per-class IoU and traversability accuracy are intentionally absent because no labeled sensor-ground-truth dataset exists.

## Progress and next steps

- **Implemented and verified:** isolated full copy; raw RayCaster-to-grid adapter on constructed hits; geometric feature and spatial affordance maps; deterministic action reasoning; four reproducible scenarios; browser demo and computed JSON/CSV/PNG exports; focused unit checks.
- **Implemented and live-validated:** the raw RayCaster adapter and GUI-to-dashboard bridge. A visible Kit run loaded CRA373 rough terrain and produced a complete 17 × 11 scanner grid. The browser endpoint is served by the same simulation process.
- **LAS research pending:** collect temporally aligned PPO observations, raw scans, state, command, 12-dimensional actions, joints, terrain ID and outcomes from the existing checkpoint. Split terrain seeds and layouts before training. Train a configurable action-sequence autoencoder on those windows; freeze its encoder to produce paired latent targets; train an affordance/state/command-to-latent mapper. Evaluate reconstruction and latent MSE, temporal consistency, held-out terrains and PCA plots. The browser must remain in the LAS-unavailable state until real checkpoints and metrics exist.
- **Closed-loop integration pending:** verify controller capability limits in simulation, validate action quality and stability offline, then compare to the original PPO baseline in headless rollouts. No decoded LAS action should be executed before those checks.
- **Unavailable:** RGB-D capture, learned visual grounding, material/friction estimates, validated climb/jump policies, route planning and calibrated action probabilities.

## Isaac Sim viewport traversability overlay

`live_affordance.py` now adds a world-aligned, noncolliding Isaac Lab `VisualizationMarkers` PointInstancer at `/World/Visuals/CRA373_Traversability` in a Kit GUI. One colored marker is updated per finite RayCaster hit. The adapter preserves the raw world XYZ of a representative hit for each 0.1 m grid cell; a 0.025 m render offset prevents z fighting. Robot movement and yaw are naturally reflected because each frame uses fresh world hits. The visualization has independent rate and style controls (`--viz-update-hz`, `--viz-point-radius-m`, `--viz-opacity`, `--viz-max-distance-m`, `--viz-green-threshold`, `--viz-yellow-threshold`, `--viz-sample-stride`); `--no-traversability-viz` disables it. Headless runs skip marker creation while continuing dashboard analysis.

This is explicitly **Heuristic Traversability Mode**. Score = `clip(1 - 0.5 * max(normalized slope, roughness, discontinuity, obstacle height), 0, 1)`, with green ≥ 0.7, yellow ≥ 0.4 and red below 0.4 by default. The normalizers are unvalidated `CapabilityHypotheses`, so neither color nor score is a learned LAS output or a calibrated success probability. Missing hits are unknown gray in the logical map but have no trustworthy 3D surface point and are omitted from the viewport. No locomotion policy, PPO reward, actuator setting or terrain collision was changed. The live rough demo defaults to the copied 48-input flat PPO actor through a read-only observation-layout adapter; rough-terrain locomotion performance is unvalidated. `--control neutral` is available for a stationary control baseline.

Validation evidence is under `validation/`: GUI and headless run logs plus targeted Isaac Sim and browser captures. GUI live classifications included green, yellow and red on the seeded random-rough terrain; all 187 rays returned finite hits, so gray remained absent in that run and is covered by the unknown-region test. The dashboard API publishes the 17 × 11 score/category grid.

Dashboard ports are now reserved before Isaac Sim starts, using a standard-library-only module. If the preferred port is occupied, the launcher tries the next ten ports and prints the actual URL. `--dashboard-port 0` uses an OS-selected port. This avoids a late `Address already in use` failure after scene startup. The socket is kept reserved until the real dashboard handler is installed after AppLauncher initialization.
