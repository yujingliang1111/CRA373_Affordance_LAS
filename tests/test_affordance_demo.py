"""Run with python -m unittest discover -s tests -v."""
import unittest
import numpy as np
from cra373.affordance.terrain_observation import TerrainObservation, from_isaac_ray_hits, yaw_from_xyzw
from cra373.affordance.terrain_affordance import estimate_affordance
from cra373.demo.scenarios import NAMES, load_scenario
from cra373.demo.pipeline import analyze

class TerrainDemoTests(unittest.TestCase):
    def test_four_recommendations(self):
        expected = dict(zip(NAMES, ("Walk", "Slow Walk", "Step Over / Climb", "Stop")))
        for name, action in expected.items():
            with self.subTest(name=name):
                result = analyze(load_scenario(name))
                self.assertEqual(result["reasoning"]["selected_action"], action)
                self.assertEqual(np.shape(result["maps"]["foothold"]), (17, 11))
                self.assertGreater(result["latency_ms"]["total"], 0)

    def test_missing_ray_hit_is_unknown_not_gap(self):
        h = np.zeros((17, 11))
        valid = np.ones_like(h, dtype=bool)
        valid[10:12, 4:7] = False
        aff = estimate_affordance(TerrainObservation(h, valid, 0.1))
        self.assertTrue(aff.unknown[10, 5])
        self.assertFalse(aff.gap[10, 5])

    def test_isaac_xyzw_yaw(self):
        yaw = yaw_from_xyzw((0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)))
        self.assertAlmostEqual(yaw, np.pi / 2)

    def test_world_hits_to_yaw_grid(self):
        # A hit 0.2 m world +y is 0.2 m forward at yaw +90 degrees.
        obs = from_isaac_ray_hits(np.array([[1.0, 2.2, 0.15]]), (1.0, 2.0, 0.32),
                                  np.pi / 2, shape=(5, 5), resolution_m=0.1)
        self.assertTrue(obs.valid[4, 2])
        self.assertAlmostEqual(obs.height_m[4, 2], 0.15)

    def test_unvalidated_skill_is_not_executable(self):
        result = analyze(load_scenario("Raised step"))["reasoning"]
        climb = next(a for a in result["actions"] if a["action"] == "Step Over / Climb")
        self.assertFalse(climb["execution_supported"])
        self.assertIsNone(next(a for a in result["actions"] if a["action"] == "Jump")["heuristic_score"])

if __name__ == "__main__":
    unittest.main()
