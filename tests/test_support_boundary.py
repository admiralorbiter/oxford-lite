"""
Unit tests for OXFORD Exploration 2B: Support Boundary & Lineage Laundering Assay
"""

import unittest

from assays import support_boundary as sb


class SupportBoundaryTests(unittest.TestCase):
    def test_independent_world_generation(self):
        w = sb.generate_independent_world("w_ind_01", seed=42, depth=3, distractors=2)
        self.assertEqual(w.world_id, "w_ind_01")
        self.assertEqual(w.mode, "INDEPENDENT")
        self.assertEqual(w.depth, 3)
        self.assertEqual(len(w.root_facts), 2)
        self.assertEqual(len(w.distractor_facts), 2)
        self.assertEqual(len(w.rules), 2)

    def test_shared_root_world_generation(self):
        w = sb.generate_shared_root_world("w_shared_01", seed=42, depth=2, distractors=1)
        self.assertEqual(w.world_id, "w_shared_01")
        self.assertEqual(w.mode, "SHARED_ROOT")
        self.assertEqual(len(w.root_facts), 1)
        self.assertEqual(len(w.intermediate_facts), 2)

    def test_laundered_echo_world_generation(self):
        w = sb.generate_laundered_echo_world("w_laundered_01", seed=42, distractors=2)
        self.assertEqual(w.world_id, "w_laundered_01")
        self.assertEqual(w.mode, "LAUNDERED_ECHO")
        self.assertEqual(len(w.root_facts), 1)  # Raw unconfirmed signal log
        self.assertEqual(len(w.intermediate_facts), 3)  # 3 derivative echo reports

    def test_adversarial_twin_isomorphism(self):
        w = sb.generate_independent_world("w_ind_01", seed=42, depth=3, distractors=2)
        twin = sb.generate_adversarial_boundary_twin(w, seed=42)
        self.assertTrue(twin.is_twin)
        self.assertEqual(twin.twin_of, "w_ind_01")
        self.assertNotEqual(w.target_entity, twin.target_entity)
        self.assertNotEqual(w.target_property, twin.target_property)
        # Check fact IDs are scrambled
        self.assertTrue(any("FACT_TWIN" in fid for fid in twin.fact_descriptions.keys()))

    def test_independent_trajectory_ground_truth(self):
        w = sb.generate_independent_world("w_ind_01", seed=42, depth=2, distractors=1)
        traj = sb.generate_boundary_trajectory(w)
        # [base, cut_r1, cut_m1, cut_both, rescue_r1, sham_dist]
        expected = ["ACTIVE", "ACTIVE", "ACTIVE", "UNKNOWN", "ACTIVE", "ACTIVE"]
        self.assertEqual(traj.ground_truth_vector, expected)

    def test_shared_root_trajectory_ground_truth(self):
        w = sb.generate_shared_root_world("w_shared_01", seed=42, depth=2, distractors=1)
        traj = sb.generate_boundary_trajectory(w)
        # [base, cut_branch_alpha, cut_shared_root, rescue_shared_root, sham_dist]
        expected = ["ACTIVE", "ACTIVE", "UNKNOWN", "ACTIVE", "ACTIVE"]
        self.assertEqual(traj.ground_truth_vector, expected)

    def test_laundered_echo_trajectory_ground_truth(self):
        w = sb.generate_laundered_echo_world("w_laundered_01", seed=42, distractors=1)
        traj = sb.generate_boundary_trajectory(w)
        # [base, cut_echo_rep1, cut_raw_origin, rescue_origin, sham_dist]
        expected = ["ACTIVE", "ACTIVE", "UNKNOWN", "ACTIVE", "ACTIVE"]
        self.assertEqual(traj.ground_truth_vector, expected)

    def test_strict_response_parser(self):
        self.assertEqual(sb.parse_response_state("ACTIVE"), "ACTIVE")
        self.assertEqual(sb.parse_response_state("UNKNOWN"), "UNKNOWN")
        self.assertEqual(sb.parse_response_state("RETRACTED"), "RETRACTED")
        self.assertEqual(sb.parse_response_state("  Active.\n"), "ACTIVE")
        self.assertEqual(sb.parse_response_state("The state is UNKNOWN"), "FORMAT_FAILURE")
        self.assertEqual(sb.parse_response_state(""), "FORMAT_FAILURE")

    def test_synthesize_boundary_corpus(self):
        corpus = sb.synthesize_boundary_corpus(seed=42)
        # 4 IND + 2 SHARED + 2 LAUNDERED = 8 worlds
        self.assertEqual(len(corpus), 8)
        self.assertIn("world", corpus[0])
        self.assertIn("twin", corpus[0])
        self.assertEqual(corpus[0]["mode"], "INDEPENDENT")


if __name__ == "__main__":
    unittest.main()
