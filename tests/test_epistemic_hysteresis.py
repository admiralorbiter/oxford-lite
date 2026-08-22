#!/usr/bin/env python3
"""Unit tests for OXFORD Exploration 3B Epistemic Hysteresis engine."""

import unittest
from assays import epistemic_hysteresis as eh


class TestEpistemicHysteresis(unittest.TestCase):
    def test_hysteresis_world_structure(self):
        world = eh.build_hysteresis_world(
            world_id="hys_test_001",
            target_entity="EXOPLANET_GLIESE_887_C",
            target_property="ATMOSPHERIC_METHANE_BIOSIGNATURE",
            domain="ASTROPHYSICS",
        )
        self.assertEqual(world.world_id, "hys_test_001")
        self.assertEqual(len(world.conditions), 5)

        cond_types = [c["condition_type"] for c in world.conditions]
        self.assertIn("STATIC", cond_types)
        self.assertIn("SEQ_NO_COMMIT", cond_types)
        self.assertIn("SEQ_COMMIT", cond_types)
        self.assertIn("PATH_REVERSAL", cond_types)
        self.assertIn("NULL_REPEAT", cond_types)

    def test_hysteresis_decomposition_metrics(self):
        # Perfect consistency case
        perfect_obs = {
            "STATIC": "ACTIVE",
            "SEQ_NO_COMMIT": "ACTIVE",
            "SEQ_COMMIT": "ACTIVE",
            "PATH_REVERSAL": "ACTIVE",
            "NULL_REPEAT": "ACTIVE",
        }
        res = eh.compute_hysteresis_decomposition(perfect_obs)
        self.assertEqual(res["H_total"], 0.0)
        self.assertEqual(res["H_sequence"], 0.0)
        self.assertEqual(res["H_commit"], 0.0)
        self.assertEqual(res["H_order"], 0.0)
        self.assertEqual(res["H_drift"], 0.0)

        # Early commitment anchoring failure case
        commit_biased_obs = {
            "STATIC": "ACTIVE",
            "SEQ_NO_COMMIT": "ACTIVE",
            "SEQ_COMMIT": "UNKNOWN",  # Anchored on retraction of A, failed to switch to D
            "PATH_REVERSAL": "ACTIVE",
            "NULL_REPEAT": "UNKNOWN",
        }
        res_biased = eh.compute_hysteresis_decomposition(commit_biased_obs)
        self.assertEqual(res_biased["H_total"], 1.0)
        self.assertEqual(res_biased["H_commit"], 1.0)
        self.assertEqual(res_biased["H_sequence"], 0.0)


if __name__ == "__main__":
    unittest.main()
