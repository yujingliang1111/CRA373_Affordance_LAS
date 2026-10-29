"""Isaac Sim viewport and synchronized affordance dashboard for CRA373.

Run from the copied project:
    /home/dog101/miniconda3/envs/env_isaaclab_2/bin/python scripts/rsl_rl/live_affordance.py

The default controller plays an existing flat PPO checkpoint on rough terrain
through a layout adapter. It does not execute reasoning recommendations or
claim validated rough-terrain locomotion.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import shutil
import sys
import threading
import traceback
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher
from cra373.demo.dashboard_port import reserve_http_server
from http.server import BaseHTTPRequestHandler

parser = argparse.ArgumentParser(description="CRA373 Isaac Sim plus live affordance dashboard")
parser.add_argument("--dashboard-port", type=int, default=7860,
                    help="Preferred dashboard port; tries the next 10 ports if occupied (0 = OS-selected)")
parser.add_argument("--dashboard-host", default="127.0.0.1")
parser.add_argument("--no-browser", action="store_true", help="Do not open the dashboard browser window")
parser.add_argument("--frames", type=int, default=0, help="Stop after N simulation frames; zero runs until the viewport closes")
parser.add_argument("--publish-every", type=int, default=5, help="Publish one dashboard frame per N simulation steps")
parser.add_argument("--terrain-scene", choices=("showcase", "rough"), default="showcase",
                    help="Three-tile rough/stairs/gap showcase or the previous all-rough terrain")
parser.add_argument("--terrain-focus", choices=("rough", "stairs", "gap"), default="stairs",
                    help="First tile scanned by the robot in showcase mode")
parser.add_argument("--showcase-camera", choices=("overview", "follow"), default="overview",
                    help="Show all three tiles or follow the robot closely")
parser.add_argument("--showcase-cycle-seconds", type=float, default=6.0,
                    help="Seconds between switching showcase tiles; 0 keeps the selected focus")
parser.add_argument("--control", choices=("flat-ppo", "neutral"), default="flat-ppo",
                    help="Existing flat PPO actor playback or neutral joint offsets")
parser.add_argument("--checkpoint", type=Path,
                    default=PROJECT_ROOT / "logs/rsl_rl/cra373_flat/2026-09-29_09-04-03/model_4200.pt",
                    help="Existing 48-input flat PPO checkpoint; used only with --control flat-ppo")
parser.add_argument("--traversability-viz", action=argparse.BooleanOptionalAction, default=True,
                    help="Show/hide heuristic traversability markers in the Isaac Sim viewport")
parser.add_argument("--viz-update-hz", type=float, default=10.0, help="Viewport marker update frequency")
parser.add_argument("--viz-point-radius-m", type=float, default=0.035, help="Marker sphere radius in metres")
parser.add_argument("--viz-opacity", type=float, default=0.78, help="Marker opacity in (0, 1]")
parser.add_argument("--viz-max-distance-m", type=float, default=5.0, help="Maximum marker distance from robot")
parser.add_argument("--viz-sample-stride", type=int, default=1, help="Draw every Nth grid row and column")
parser.add_argument("--viz-green-threshold", type=float, default=0.7, help="Minimum heuristic score for green")
parser.add_argument("--viz-yellow-threshold", type=float, default=0.4, help="Minimum heuristic score for yellow")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.publish_every < 1 or args.frames < 0:
    parser.error("--publish-every must be positive and --frames must be nonnegative")
if not 0 <= args.dashboard_port <= 65535:
    parser.error("--dashboard-port must be 0..65535")
if args.showcase_cycle_seconds < 0:
    parser.error("--showcase-cycle-seconds must be nonnegative")
class _PendingDashboardHandler(BaseHTTPRequestHandler):
    """Used only while the socket is reserved before the real handler loads."""
    def do_GET(self):
        self.send_error(503, "Isaac Sim is starting")

try:
    reserved_dashboard = reserve_http_server(
        args.dashboard_host, args.dashboard_port, _PendingDashboardHandler, fallback_ports=10
    )
except OSError as exc:
    parser.error(f"Cannot reserve a dashboard port starting at {args.dashboard_port}: {exc}")
actual_dashboard_port = reserved_dashboard.server_address[1]
if args.dashboard_port == 0:
    print(f"[LIVE] Dashboard selected available port {actual_dashboard_port}", flush=True)
elif actual_dashboard_port != args.dashboard_port:
    print(f"[LIVE] Dashboard port {args.dashboard_port} is busy; using {actual_dashboard_port}", flush=True)

# Kit opens a visible Isaac Sim viewport when no headless visualizer was selected.
if args.visualizer is None and not args.headless and args.livestream in (-1, 0):
    args.visualizer = ["kit"]
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import cra373.envs  # register CRA373-Rough-v0
from cra373.envs.cra373_env_cfg import CRA373RoughEnvCfg
from cra373.affordance.terrain_observation import from_isaac_ray_hits, yaw_from_xyzw
from cra373.demo.pipeline import analyze
from cra373.demo.demo_app import Handler, publish_live
from cra373.affordance.terrain_affordance import estimate_affordance
from cra373.affordance.traversability_viz import TraversabilityVizConfig, make_marker_frame
from cra373.demo.viewport_traversability import ViewportTraversability
from cra373.demo.flat_policy import FlatPPOPolicy
from cra373.demo.showcase_terrain import FEATURES, configure_live_terrain, set_showcase_focus


def _to_numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def main() -> None:
    viz_cfg = TraversabilityVizConfig(
        green_threshold=args.viz_green_threshold,
        yellow_threshold=args.viz_yellow_threshold,
        max_distance_m=args.viz_max_distance_m,
        point_radius_m=args.viz_point_radius_m,
        opacity=args.viz_opacity,
        update_hz=args.viz_update_hz,
        sample_stride=args.viz_sample_stride,
    )
    viewport_enabled = args.traversability_viz and not args.headless and "kit" in (args.visualizer or [])
    if args.traversability_viz and not viewport_enabled:
        print("[LIVE] Viewport markers disabled: no Kit GUI; dashboard analysis remains active", flush=True)
    cfg = CRA373RoughEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 3.0
    cfg.seed = 373
    configure_live_terrain(cfg, args.terrain_scene)
    if args.terrain_scene == "showcase" and args.showcase_camera == "overview":
        cfg.viewer.eye = (7.0, 9.0, 10.0)
        cfg.viewer.lookat = (0.0, 1.5, 0.2)
        cfg.viewer.origin_type = "world"
    else:
        cfg.viewer.eye = (3.0, 3.0, 2.2)
        cfg.viewer.lookat = (0.0, 0.0, 0.25)
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"

    env = None
    server = reserved_dashboard
    viewport = None
    policy = None
    try:
        env = gym.make("CRA373-Rough-v0", cfg=cfg)
        robot_prim = env.unwrapped.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if not robot_prim.IsValid():
            raise RuntimeError("CRA373 robot prim was not loaded in Isaac Sim")
        if "height_scanner" not in env.unwrapped.scene.sensors:
            raise RuntimeError("CRA373 rough environment has no height_scanner")
        server.RequestHandlerClass = Handler
        threading.Thread(target=server.serve_forever, daemon=True, name="affordance-dashboard").start()
        url = f"http://{args.dashboard_host}:{actual_dashboard_port}/?mode=live"
        print(f"[LIVE] Isaac Sim viewport and terrain dashboard: {url}", flush=True)
        if not args.no_browser:
            try:
                browser = next((shutil.which(name) for name in ("google-chrome", "chromium", "chromium-browser", "firefox") if shutil.which(name)), None)
                command = ([browser, "--new-window", url] if browser else ["xdg-open", url])
                subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError as exc:
                print(f"[LIVE] Browser could not open automatically: {exc}. Open {url}", flush=True)
        active_feature = args.terrain_focus if args.terrain_scene == "showcase" else "rough"
        if args.terrain_scene == "showcase":
            set_showcase_focus(env, active_feature)
            print("[LIVE] Showcase tiles: rough | stairs | gap; "
                  f"initial scan: {active_feature}", flush=True)
        with torch.inference_mode():
            obs, _ = env.reset()
        if args.control == "flat-ppo":
            policy = FlatPPOPolicy(args.checkpoint, env.unwrapped.device)
            print(f"[LIVE] Existing flat PPO actor: {policy.checkpoint} "
                  "(rough-terrain transfer is unvalidated)", flush=True)
        else:
            print("[LIVE] Neutral joint-offset control", flush=True)
        if viewport_enabled:
            viewport = ViewportTraversability(viz_cfg)
            print(f"[LIVE] Heuristic traversability overlay enabled at {viz_cfg.update_hz:g} Hz "
                  "(green/yellow/red; gray unknown has no surface hit and is omitted)", flush=True)
        step = 0
        next_viz_time_s = 0.0
        next_cycle_time_s = args.showcase_cycle_seconds
        showcase_switch_pending = False
        while simulation_app.is_running() and (args.frames == 0 or step < args.frames):
            if (args.terrain_scene == "showcase" and args.showcase_cycle_seconds > 0
                    and step * env.unwrapped.step_dt >= next_cycle_time_s):
                active_feature = FEATURES[(FEATURES.index(active_feature) + 1) % len(FEATURES)]
                set_showcase_focus(env, active_feature)
                with torch.inference_mode():
                    obs, _ = env.reset()
                next_cycle_time_s += args.showcase_cycle_seconds
                showcase_switch_pending = True
                print(f"[LIVE] Showcase scan switched to {active_feature}", flush=True)
            # Affordance recommendations never enter the control stream.
            with torch.inference_mode():
                actions = policy(obs["policy"]) if policy is not None else torch.zeros((1, 12), device=env.unwrapped.device)
                obs, _, _, _, _ = env.step(actions)
            step += 1
            sim_time_s = step * env.unwrapped.step_dt
            publish_due = step % args.publish_every == 0
            traversability_due = sim_time_s + 1e-9 >= next_viz_time_s
            if not (publish_due or traversability_due):
                continue
            robot = env.unwrapped.scene["robot"]
            scanner = env.unwrapped.scene.sensors["height_scanner"]
            hits = _to_numpy(scanner.data.ray_hits_w.torch[0])
            position = tuple(_to_numpy(robot.data.root_pos_w.torch[0]).tolist())
            yaw = yaw_from_xyzw(_to_numpy(robot.data.root_quat_w.torch[0]))
            velocity = tuple(_to_numpy(robot.data.root_lin_vel_b.torch[0]).tolist())
            observation = from_isaac_ray_hits(
                hits, position, yaw, base_velocity_b_mps=velocity,
                timestamp_s=sim_time_s,
            )
            observation.metadata["live_step"] = step
            observation.metadata["terrain_scene"] = args.terrain_scene
            observation.metadata["terrain_feature"] = active_feature
            if not observation.valid.any():
                if viewport is not None:
                    viewport.close()
                continue
            affordance = estimate_affordance(observation)
            frame = make_marker_frame(observation, affordance, viz_cfg)
            if traversability_due:
                if viewport is not None:
                    viewport.update(frame)
                next_viz_time_s = sim_time_s + 1.0 / viz_cfg.update_hz
            if publish_due:
                payload = analyze(observation)
                payload["traversability"] = {
                    "mode": "Heuristic Traversability Mode",
                    "green_threshold": viz_cfg.green_threshold,
                    "yellow_threshold": viz_cfg.yellow_threshold,
                    "categories": frame.grid_categories.tolist(),
                    "scores": [[float(v) if np.isfinite(v) else None for v in row] for row in frame.score],
                    "unknown_without_surface_hit": frame.undrawn_unknown_count,
                }
                publish_live(payload)
                if step == args.publish_every or step % (args.publish_every * 50) == 0 or showcase_switch_pending:
                    print(f"[LIVE] step={step} valid={observation.valid.sum()}/{observation.valid.size} "
                          f"terrain={active_feature} recommendation={payload['reasoning']['selected_action']} "
                          f"markers={len(frame.positions_w_m) if viewport is not None else 0} "
                          f"heuristic_classes={np.bincount(frame.grid_categories.ravel(), minlength=4).tolist()}", flush=True)
                    showcase_switch_pending = False
    except BaseException as exc:
        print(f"[LIVE ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        if viewport is not None:
            viewport.close()
        if server is not None:
            server.shutdown()
            server.server_close()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
