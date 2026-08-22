#!/usr/bin/env python3
"""Unit tests for OXFORD Exploration 3A Support Acquisition engine."""

import unittest
from assays import support_acquisition as sa


class TestSupportAcquisition(unittest.TestCase):
    def test_candidate_graph_hypotheses_evaluation(self):
        hyps = sa.get_standard_candidate_hypotheses(["A", "B", "C", "D"])
        g_gt = next(h for h in hyps if h.name == "G_ground_truth")
        g_naive = next(h for h in hyps if h.name == "G_naive_surface")

        # When A and D are active
        self.assertEqual(g_gt.evaluate_state({"A", "B", "C", "D"}), "ACTIVE")
        # When only B and C are active (A and D retracted)
        self.assertEqual(g_gt.evaluate_state({"B", "C"}), "UNKNOWN")
        self.assertEqual(g_naive.evaluate_state({"B", "C"}), "ACTIVE")

    def test_identifiability_compiler(self):
        hyps = sa.get_standard_candidate_hypotheses(["A", "B", "C", "D"])
        res = sa.compute_identifiability_codewords(hyps, ["A", "B", "C", "D"])

        self.assertIn("G_ground_truth", res["codeword_matrix"])
        self.assertIn("G_naive_surface", res["codeword_matrix"])

        # Ground truth and naive surface should produce different codewords
        gt_code = res["codeword_matrix"]["G_ground_truth"]
        naive_code = res["codeword_matrix"]["G_naive_surface"]
        self.assertNotEqual(gt_code, naive_code)

        # Distinguishable pairs should be positive
        self.assertGreater(res["distinguishable_pair_count"], 0)

    def test_world_synthesis_and_renderers(self):
        world = sa.synthesize_acquisition_world(
            world_id="acq_test_001",
            domain="ASTROPHYSICS",
            echo_count=2,
            seed=42,
        )
        self.assertEqual(world.world_id, "acq_test_001")
        self.assertEqual(len(world.sources), 4)  # A + B, C + D
        self.assertIn(world.target_entity, world.synthetic_prose)
        self.assertIn(world.target_entity, world.naturalistic_prose)

        # Verify trajectory compilation
        traj_synth = sa.compile_acquisition_trajectory(world, stratum="SYNTHETIC")
        traj_nat = sa.compile_acquisition_trajectory(world, stratum="NATURALISTIC")

        self.assertEqual(len(traj_synth), 6)
        self.assertEqual(len(traj_nat), 6)
        self.assertEqual(traj_synth[0].expected_state, "ACTIVE")
        self.assertEqual(traj_synth[3].expected_state, "UNKNOWN")  # complete cut


if __name__ == "__main__":
    unittest.main()
