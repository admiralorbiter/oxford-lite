"""
Unit tests for OXFORD Exploration 2B.1: Formal Graph Support Boundary & Lineage Laundering Assay
"""

import unittest

from assays import support_boundary as sb


class SupportBoundaryTests(unittest.TestCase):
    def test_independent_world_generation_formal_horn(self):
        w = sb.generate_independent_world("w_ind_01", seed=42, depth=3, distractors=2)
        self.assertEqual(w.world_id, "w_ind_01")
        self.assertEqual(w.mode, "INDEPENDENT")
        self.assertEqual(w.depth, 3)
        self.assertEqual(len(w.root_facts), 2)
        self.assertEqual(len(w.distractor_facts), 2)
        # Depth 3 has (2 relays + 1 auth) * 2 paths = 6 intermediate facts
        self.assertEqual(len(w.intermediate_facts), 6)
        # 6 formal step-by-step Horn rules
        self.assertEqual(len(w.rules), 6)

    def test_shared_root_world_generation_multi_hop(self):
        w_d2 = sb.generate_shared_root_world("w_shared_d2", seed=42, depth=2, distractors=1)
        w_d4 = sb.generate_shared_root_world("w_shared_d4", seed=42, depth=4, distractors=1)
        self.assertEqual(w_d2.depth, 2)
        self.assertEqual(w_d4.depth, 4)
        # Depth 4 has more intermediate steps than Depth 2
        self.assertGreater(len(w_d4.intermediate_facts), len(w_d2.intermediate_facts))

    def test_laundered_echo_world_explicitly_valid(self):
        w = sb.generate_laundered_echo_world("w_laundered_01", seed=42, distractors=2)
        self.assertEqual(w.world_id, "w_laundered_01")
        self.assertEqual(w.mode, "LAUNDERED_ECHO")
        self.assertEqual(len(w.root_facts), 1)
        self.assertEqual(len(w.intermediate_facts), 3)
        # Primary log is explicitly verified in baseline
        self.assertIn("verified", w.fact_descriptions[w.root_facts[0]].lower())

    def test_ast_conjunction_inversion(self):
        rule = "If entity is connected to node X AND node X forwards connection to node Y, then entity is connected to node Y."
        inverted = sb.invert_rule_conjunctions(rule)
        self.assertTrue(inverted.startswith("If node X forwards connection to node Y AND entity is connected to node X, then entity is connected to node Y."))

    def test_adversarial_twin_isomorphism(self):
        w = sb.generate_independent_world("w_ind_01", seed=42, depth=3, distractors=2)
        twin = sb.generate_adversarial_boundary_twin(w, seed=42)
        self.assertTrue(twin.is_twin)
        self.assertEqual(twin.twin_of, "w_ind_01")
        self.assertNotEqual(w.target_entity, twin.target_entity)
        self.assertNotEqual(w.target_property, twin.target_property)
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

    def test_synthesize_boundary_corpus_balanced(self):
        corpus = sb.synthesize_boundary_corpus(seed=42)
        # 8 IND (2 per depth d=2..5) + 2 SHARED (d=2,4) + 2 LAUNDERED = 12 worlds
        self.assertEqual(len(corpus), 12)
        self.assertIn("world", corpus[0])
        self.assertIn("twin", corpus[0])
        ind_worlds = [c for c in corpus if c["mode"] == "INDEPENDENT"]
        self.assertEqual(len(ind_worlds), 8)


if __name__ == "__main__":
    unittest.main()
