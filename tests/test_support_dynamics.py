import unittest

from assays import support_dynamics as sd


class SupportDynamicsTests(unittest.TestCase):
    def test_world_generation(self):
        world = sd.generate_support_world("w_001", seed=42)
        self.assertEqual(world.world_id, "w_001")
        self.assertEqual(len(world.path_1_facts), 2)
        self.assertEqual(len(world.path_2_facts), 2)
        self.assertTrue(world.distractor_fact.startswith("FACT_E_"))
        self.assertEqual(len(world.fact_descriptions), 5)

    def test_isomorphic_twin(self):
        world = sd.generate_support_world("w_001", seed=42)
        twin = sd.generate_isomorphic_twin(world, seed=42)
        self.assertTrue(twin.is_twin)
        self.assertEqual(twin.twin_of, "w_001")
        self.assertNotEqual(world.target_entity, twin.target_entity)

    def test_trajectory_ground_truth(self):
        world = sd.generate_support_world("w_001", seed=42)
        traj = sd.generate_world_trajectory(world)
        self.assertEqual(len(traj.interventions), 8)
        
        # Expected ground truth:
        # [base, -A, -C, -AC, -AB, -ABC, rescue_A, sham_E]
        # [ACTIVE, ACTIVE, ACTIVE, UNKNOWN, ACTIVE, UNKNOWN, ACTIVE, ACTIVE]
        expected = [
            "ACTIVE",   # base
            "ACTIVE",   # -A (C,D survives)
            "ACTIVE",   # -C (A,B survives)
            "UNKNOWN",  # -AC (both paths cut)
            "ACTIVE",   # -AB (C,D survives)
            "UNKNOWN",  # -ABC (both paths cut)
            "ACTIVE",   # rescue (+A after cut)
            "ACTIVE",   # sham (-E unrelated)
        ]
        self.assertEqual(traj.ground_truth_vector, expected)

    def test_sham_condition_flags(self):
        world = sd.generate_support_world("w_001", seed=42)
        traj = sd.generate_world_trajectory(world)
        sham_items = [item for item in traj.interventions if item.is_sham]
        self.assertEqual(len(sham_items), 1)
        self.assertEqual(sham_items[0].condition_id, "c08_sham_e")
        self.assertEqual(sham_items[0].expected_state, "ACTIVE")

    def test_parse_response_state(self):
        self.assertEqual(sd.parse_response_state("ACTIVE"), "ACTIVE")
        self.assertEqual(sd.parse_response_state("UNKNOWN"), "UNKNOWN")
        self.assertEqual(sd.parse_response_state("RETRACTED"), "RETRACTED")
        self.assertEqual(sd.parse_response_state("  Active.\n"), "ACTIVE")
        self.assertEqual(sd.parse_response_state("The state is UNKNOWN"), "UNKNOWN")
        self.assertEqual(sd.parse_response_state(""), "ERROR")

    def test_trajectory_distance_and_stability(self):
        vec_1 = ["ACTIVE", "ACTIVE", "ACTIVE", "UNKNOWN", "ACTIVE", "UNKNOWN", "ACTIVE", "ACTIVE"]
        vec_2 = ["ACTIVE", "ACTIVE", "ACTIVE", "UNKNOWN", "ACTIVE", "UNKNOWN", "ACTIVE", "ACTIVE"]
        vec_3 = ["ACTIVE", "ACTIVE", "UNKNOWN", "UNKNOWN", "ACTIVE", "UNKNOWN", "ACTIVE", "ACTIVE"] # 1 diff
        
        # Perfect match
        self.assertEqual(sd.trajectory_distance(vec_1, vec_2), 0.0)
        self.assertEqual(sd.trajectory_accuracy(vec_1, vec_2), 1.0)
        
        # 1 mismatch out of 8
        self.assertEqual(sd.trajectory_distance(vec_1, vec_3), 0.125)
        self.assertEqual(sd.trajectory_accuracy(vec_3, vec_1), 0.875)

    def test_synthesize_dynamics_corpus(self):
        corpus = sd.synthesize_dynamics_corpus(count=5, seed=100)
        self.assertEqual(len(corpus), 5)
        self.assertIn("world", corpus[0])
        self.assertIn("twin", corpus[0])
        self.assertEqual(len(corpus[0]["trajectory"]), 8)


if __name__ == "__main__":
    unittest.main()
