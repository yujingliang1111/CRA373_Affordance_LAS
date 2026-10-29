"""Isaac Lab PointInstancer backend for the CRA373 heuristic overlay."""
from __future__ import annotations

import numpy as np
from cra373.affordance.traversability_viz import MarkerFrame, TraversabilityVizConfig


class ViewportTraversability:
    def __init__(self, config: TraversabilityVizConfig):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        colors = {
            "green": (0.08, 0.85, 0.20),
            "yellow": (1.0, 0.72, 0.06),
            "red": (0.95, 0.10, 0.09),
            "gray_unknown": (0.55, 0.58, 0.63),
        }
        markers = {
            name: sim_utils.SphereCfg(
                radius=config.point_radius_m,
                collision_props=None,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=config.opacity),
            )
            for name, color in colors.items()
        }
        self._markers = VisualizationMarkers(
            VisualizationMarkersCfg(prim_path="/World/Visuals/CRA373_Traversability", markers=markers)
        )
        self._markers.set_visibility(False)

    def update(self, frame: MarkerFrame) -> None:
        if len(frame.positions_w_m) == 0:
            self._markers.set_visibility(False)
            return
        self._markers.set_visibility(True)
        self._markers.visualize(
            translations=np.ascontiguousarray(frame.positions_w_m),
            marker_indices=np.ascontiguousarray(frame.categories),
        )

    def close(self) -> None:
        self._markers.set_visibility(False)
