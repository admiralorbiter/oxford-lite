#!/usr/bin/env python3
"""Unit tests for OXFORD Lineage Calibration and LORO Engine."""

import unittest
from assays import lineage_calibration as lc


class TestLineageCalibration(unittest.TestCase):
    def setUp(self):
        self.holdout_sample = [
            {
                "world": {"world_id": "w01", "echo_count": 2, "domain": "ASTROPHYSICS"},
                "synthetic_trajectory": [
                    {"condition_id": "c01_base", "prompt_text": "P1"},
                    {"condition_id": "c02_retract_primary_root", "prompt_text": "P2"},
                    {"condition_id": "c04_complete_root_cut", "prompt_text": "P4"},
                ],
                "naturalistic_trajectory": [
                    {"condition_id": "c01_base", "prompt_text": "P1"},
                    {"condition_id": "c02_retract_primary_root", "prompt_text": "P2"},
                    {"condition_id": "c04_complete_root_cut", "prompt_text": "P4"},
                ],
                "ground_truth": ["ACTIVE", "ACTIVE", "UNKNOWN"],
            }
        ]

    def test_compute_four_channel_vector(self):
        attempts = {
            "w01_SYNTHETIC_c01_base": {"ok": True, "probe_id": "w01_SYNTHETIC_c01_base", "response_json": {"choices": [{"message": {"content": "ACTIVE"}}]}},
            "w01_SYNTHETIC_c02_retract_primary_root": {"ok": True, "probe_id": "w01_SYNTHETIC_c02_retract_primary_root", "response_json": {"choices": [{"message": {"content": "ACTIVE"}}]}},
            "w01_SYNTHETIC_c04_complete_root_cut": {"ok": True, "probe_id": "w01_SYNTHETIC_c04_complete_root_cut", "response_json": {"choices": [{"message": {"content": "UNKNOWN"}}]}},
            "w01_NATURALISTIC_c01_base": {"ok": True, "probe_id": "w01_NATURALISTIC_c01_base", "response_json": {"choices": [{"message": {"content": "ACTIVE"}}]}},
            "w01_NATURALISTIC_c02_retract_primary_root": {"ok": True, "probe_id": "w01_NATURALISTIC_c02_retract_primary_root", "response_json": {"choices": [{"message": {"content": "ACTIVE"}}]}},
            "w01_NATURALISTIC_c04_complete_root_cut": {"ok": True, "probe_id": "w01_NATURALISTIC_c04_complete_root_cut", "response_json": {"choices": [{"message": {"content": "RETRACTED"}}]}},
        }
        prof = lc.compute_four_channel_vector("ModelA", attempts, self.holdout_sample, tokenizer_family="GLM")
        self.assertEqual(prof["model_name"], "ModelA")
        self.assertEqual(prof["structural"]["tokenizer_family"], "GLM")
        self.assertEqual(prof["overall_accuracy"][:2], (5, 6))
        self.assertEqual(prof["cognitive"]["c2_root_retention"][:2], (2, 2))
        self.assertEqual(prof["calibration"]["F_false_standard"][:2], (1, 2))
        self.assertEqual(prof["surface"]["renderer_stability"][:2], (2, 3))

    def test_compute_pairwise_distances(self):
        prof_a = {
            "decisions": {"p1": "ACTIVE", "p2": "ACTIVE", "p3": "UNKNOWN"},
            "contracts": {"p1": 1, "p2": 1, "p3": 1},
            "flip_masks": {("w01", "c01"): 0},
        }
        prof_b = {
            "decisions": {"p1": "ACTIVE", "p2": "UNKNOWN", "p3": "UNKNOWN"},
            "contracts": {"p1": 1, "p2": 0, "p3": 1},
            "flip_masks": {("w01", "c01"): 1},
        }
        dists = lc.compute_pairwise_distances(prof_a, prof_b, self.holdout_sample)
        self.assertEqual(dists["n_shared"], 3)
        self.assertEqual(dists["D_total"][:2], (1, 3))
        self.assertEqual(dists["D_contract"][:2], (1, 3))
        self.assertEqual(dists["D_render"][:2], (1, 1))

    def test_evaluate_loro_clustering(self):
        profiles = {
            "M1_rel": {
                "decisions": {"p1": "ACTIVE", "p2": "ACTIVE"},
                "contracts": {"p1": 1, "p2": 1},
                "flip_masks": {("w01", "c01"): 0},
            },
            "M2_rel": {
                "decisions": {"p1": "ACTIVE", "p2": "ACTIVE"},
                "contracts": {"p1": 1, "p2": 1},
                "flip_masks": {("w01", "c01"): 0},
            },
            "M_ctrl": {
                "decisions": {"p1": "UNKNOWN", "p2": "UNKNOWN"},
                "contracts": {"p1": 0, "p2": 0},
                "flip_masks": {("w01", "c01"): 1},
            },
        }
        known_fams = {"FamilyA": ["M1_rel", "M2_rel"]}
    def test_compute_informative_channel_vector(self):
        prof_base = {
            "model_name": "GLM-5.2",
            "decisions": {"p1": "ACTIVE", "p2": "ACTIVE", "p3": "UNKNOWN", "p4": "UNKNOWN"},
            "contracts": {"p1": 1, "p2": 1, "p3": 1, "p4": 1},
            "flip_masks": {("w01", "c01"): 0},
        }
        prof_desc = {
            "model_name": "GLM-5.3",
            "decisions": {"p1": "ACTIVE", "p2": "ACTIVE", "p3": "RETRACTED", "p4": "RETRACTED"},
            "contracts": {"p1": 1, "p2": 1, "p3": 1, "p4": 1},
            "flip_masks": {("w01", "c01"): 0},
        }
        prof_target = {
            "model_name": "Ox Alpha",
            "decisions": {"p1": "ACTIVE", "p2": "ACTIVE", "p3": "RETRACTED", "p4": "RETRACTED"},
            "contracts": {"p1": 1, "p2": 1, "p3": 1, "p4": 1},
            "flip_masks": {("w01", "c01"): 0},
        }
        res = lc.compute_informative_channel_vector(prof_base, prof_desc, prof_target, self.holdout_sample)
        self.assertEqual(res["base_model"], "GLM-5.2")
        self.assertEqual(res["descendant_model"], "GLM-5.3")
        self.assertEqual(res["target_model"], "Ox Alpha")
        ch_tot = res["channel_analysis"]["D_total"]
        self.assertTrue(ch_tot["is_informative"])
        self.assertEqual(ch_tot["n_discordant_cells"], 2)
        self.assertEqual(ch_tot["N_match_desc"], 2)
        self.assertEqual(ch_tot["N_match_base"], 0)
        self.assertEqual(ch_tot["Match_desc"], 1.0)
        self.assertEqual(ch_tot["Branch_Index_Bk"], 1.0)
        self.assertEqual(ch_tot["target_placement"], "FAVORS_DESCENDANT")


if __name__ == "__main__":
    unittest.main()
