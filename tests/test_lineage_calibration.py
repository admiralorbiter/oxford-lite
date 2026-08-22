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
        self.assertEqual(prof["cognitive"]["semantic_acc"][:2], (5, 6))
        self.assertEqual(prof["cognitive"]["c2_root_retention"][:2], (2, 2))
        self.assertEqual(prof["calibration"]["false_falsification_standard"][:2], (1, 2))
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
        res = lc.evaluate_loro_clustering(profiles, known_fams, self.holdout_sample)
        self.assertEqual(res["total_loro_tests"], 2)
        self.assertEqual(res["successful_recoveries"], 2)
        self.assertEqual(res["loro_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
