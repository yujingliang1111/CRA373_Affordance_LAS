"""Geometry, color and world-alignment checks without an Isaac Sim dependency."""
import unittest
import numpy as np

from cra373.affordance.terrain_observation import TerrainObservation, from_isaac_ray_hits
from cra373.affordance.terrain_affordance import estimate_affordance
from cra373.affordance.traversability_viz import (
    GREEN, YELLOW, RED, GRAY, TraversabilityVizConfig,
    heuristic_traversability, make_marker_frame,
)


def observation_from_heights(heights, valid=None):
    heights = np.asarray(heights, dtype=float)
    rows, cols = heights.shape
    x, y = np.meshgrid(np.arange(rows) * 0.1, np.arange(cols) * 0.1, indexing="ij")
    points = np.stack((x, y, heights), axis=-1)
    if valid is None:
        valid = np.ones_like(heights, dtype=bool)
    points[~valid] = np.nan
    return TerrainObservation(heights, valid, 0.1, hit_points_w_m=points)


class TraversabilityVizTests(unittest.TestCase):
    def test_flat_green_and_real_world_positions(self):
        obs = observation_from_heights(np.zeros((5, 5)))
        frame = make_marker_frame(obs, estimate_affordance(obs))
        self.assertEqual(len(frame.positions_w_m), 25)
        self.assertTrue(np.all(frame.categories == GREEN))
        np.testing.assert_allclose(frame.positions_w_m[0], (0, 0, 0.025), atol=1e-7)

    def test_moderate_step_yellow_and_severe_step_red(self):
        moderate = np.zeros((7, 7))
        moderate[3:] = 0.05
        obs = observation_from_heights(moderate)
        _, colors = heuristic_traversability(obs, estimate_affordance(obs))
        self.assertEqual(colors[3, 3], YELLOW)
        severe = np.zeros((7, 7))
        severe[3:] = 0.25
        obs = observation_from_heights(severe)
        _, colors = heuristic_traversability(obs, estimate_affordance(obs))
        self.assertEqual(colors[3, 3], RED)

    def test_steep_slope_red(self):
        h = np.repeat((np.arange(7) * 0.06)[:, None], 7, axis=1)
        obs = observation_from_heights(h)
        _, colors = heuristic_traversability(obs, estimate_affordance(obs))
        self.assertEqual(colors[3, 3], RED)

    def test_unknown_no_invented_ground_marker(self):
        valid = np.ones((5, 5), dtype=bool)
        valid[2, 2] = False
        obs = observation_from_heights(np.zeros((5, 5)), valid)
        aff = estimate_affordance(obs)
        frame = make_marker_frame(obs, aff)
        self.assertEqual(frame.grid_categories[2, 2], GRAY)
        self.assertEqual(frame.grid_categories[2, 1], YELLOW)
        self.assertEqual(frame.unknown_count, 1)
        self.assertEqual(frame.undrawn_unknown_count, 1)
        self.assertEqual(len(frame.positions_w_m), 24)

    def test_moving_yaw_frame_keeps_raw_hit_position(self):
        yaw = np.pi / 2
        hit = np.array([[1.013, 2.198, 0.12]])
        obs = from_isaac_ray_hits(hit, (1.0, 2.0, 0.32), yaw,
                                  shape=(5, 5), resolution_m=0.1)
        frame = make_marker_frame(obs, estimate_affordance(obs))
        self.assertEqual(len(frame.positions_w_m), 1)
        np.testing.assert_allclose(frame.positions_w_m[0], hit[0] + [0, 0, 0.025], atol=1e-6)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            TraversabilityVizConfig(green_threshold=0.3, yellow_threshold=0.4)


if __name__ == "__main__":
    unittest.main()
