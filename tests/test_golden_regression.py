#!/usr/bin/env python3
"""Golden regression test suite for OXFORD Lineage Calibration.

Locks and asserts the exact empirical values for Ox Alpha, GPT-4o-mini,
and Laguna-S 2.1 across all 4 channels, pairwise shared distances,
and the common-cells intersection.
"""

import json
import unittest
from pathlib import Path
from assays import lineage_calibration as lc

ROOT = Path(__file__).resolve().parent.parent


class TestGoldenRunRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.holdout_file = ROOT / "worlds" / "holdout" / "support_acquisition_holdout.json"
        cls.holdouts = json.loads(cls.holdout_file.read_text(encoding="utf-8"))

        cls.ox_dir = ROOT / "runs" / "20260821-204102-acquisition"
        cls.gpt_dir = ROOT / "runs" / "20260821-211156-acquisition"
        cls.laguna_dir = ROOT / "runs" / "20260821-214232-acquisition"

        cls.ox_attempts = lc.load_run_attempts(cls.ox_dir)
        cls.gpt_attempts = lc.load_run_attempts(cls.gpt_dir)
        cls.laguna_attempts = lc.load_run_attempts(cls.laguna_dir)

        cls.ox_prof = lc.compute_four_channel_vector("Ox Alpha", cls.ox_attempts, cls.holdouts, tokenizer_family="GLM")
        cls.gpt_prof = lc.compute_four_channel_vector("GPT-4o-mini", cls.gpt_attempts, cls.holdouts, tokenizer_family="UNKNOWN")
        cls.laguna_prof = lc.compute_four_channel_vector("Laguna-S", cls.laguna_attempts, cls.holdouts, tokenizer_family="UNKNOWN")

    def test_ox_alpha_golden_metrics(self):
        cog = self.ox_prof["cognitive"]
        sur = self.ox_prof["surface"]
        cal = self.ox_prof["calibration"]
        
        # 144 decisions evaluated
        self.assertEqual(self.ox_prof["total_evaluated"], 144)
        # Strict contract = 89 / 144 = 61.8%
        self.assertEqual(sur["contract_adherence"][:2], (89, 144))
        # Renderer stability = 64 / 72 = 88.9%
        self.assertEqual(sur["renderer_stability"][:2], (64, 72))
        # C2 Retention = 23 / 24 = 95.8%
        self.assertEqual(cog["c2_root_retention"][:2], (23, 24))
        # Broad Support Abandonment F_abandon- = 0 / 96 = 0.0%
        self.assertEqual(cog["F_abandon_minus"][:2], (0, 96))
        self.assertEqual(cog["F_A_to_U"][:2], (0, 96))
        self.assertEqual(cog["F_A_to_R"][:2], (0, 96))
        # False Survival F+ = 0 / 24 = 0.0%
        self.assertEqual(cog["F_plus_survival"][:2], (0, 24))
        # Standard False Falsification = 7 / 24 = 29.2%
        self.assertEqual(cal["F_false_standard"][:2], (7, 24))

    def test_gpt_4o_mini_golden_metrics(self):
        cog = self.gpt_prof["cognitive"]
        sur = self.gpt_prof["surface"]
        cal = self.gpt_prof["calibration"]

        # 127 decisions evaluated
        self.assertEqual(self.gpt_prof["total_evaluated"], 127)
        # Strict contract = 100 / 127 = 78.7%
        self.assertEqual(sur["contract_adherence"][:2], (100, 127))
        # Renderer stability = 42 / 61 = 68.9%
        self.assertEqual(sur["renderer_stability"][:2], (42, 61))
        # C2 Retention = 4 / 21 = 19.0%
        self.assertEqual(cog["c2_root_retention"][:2], (4, 21))
        # Broad Support Abandonment F_abandon- = 22 / 84 = 26.2%
        self.assertEqual(cog["F_abandon_minus"][:2], (22, 84))
        self.assertEqual(cog["F_A_to_U"][:2], (12, 84))
        self.assertEqual(cog["F_A_to_R"][:2], (10, 84))
        # False Survival F+ = 2 / 21 = 9.5%
        self.assertEqual(cog["F_plus_survival"][:2], (2, 21))
        # Standard False Falsification = 3 / 21 = 14.3%
        self.assertEqual(cal["F_false_standard"][:2], (3, 21))

    def test_pairwise_shared_distances(self):
        # Ox vs GPT on 127 shared cells
        ox_gpt = lc.compute_pairwise_distances(self.ox_prof, self.gpt_prof, self.holdouts)
        self.assertEqual(ox_gpt["n_shared"], 127)
        self.assertEqual(ox_gpt["D_total"][:2], (35, 127))
        self.assertEqual(ox_gpt["D_acq"][:2], (24, 84))
        self.assertEqual(ox_gpt["D_contract"][:2], (41, 127))

        # Ox vs Laguna on 45 shared cells
        ox_laguna = lc.compute_pairwise_distances(self.ox_prof, self.laguna_prof, self.holdouts)
        self.assertEqual(ox_laguna["n_shared"], 45)
        self.assertEqual(ox_laguna["D_total"][:2], (5, 45))
        self.assertEqual(ox_laguna["D_acq"][:2], (3, 28))
        self.assertEqual(ox_laguna["D_contract"][:2], (14, 45))

        # GPT vs Laguna on 45 shared cells
        gpt_laguna = lc.compute_pairwise_distances(self.gpt_prof, self.laguna_prof, self.holdouts)
        self.assertEqual(gpt_laguna["n_shared"], 45)
        self.assertEqual(gpt_laguna["D_total"][:2], (12, 45))
        self.assertEqual(gpt_laguna["D_acq"][:2], (8, 28))
        self.assertEqual(gpt_laguna["D_contract"][:2], (1, 45))

    def test_common_cells_matrix(self):
        profiles = {
            "Ox Alpha": self.ox_prof,
            "GPT-4o-mini": self.gpt_prof,
            "Laguna-S": self.laguna_prof,
        }
        res = lc.compute_common_cells_matrix(profiles, self.holdouts)
        self.assertEqual(res["n_common_cells"], 45)
        
        m = res["matrix"]
        # Ox <-> GPT on 45 common cells
        self.assertEqual(m[("Ox Alpha", "GPT-4o-mini")]["D_total"][:2], (15, 45))
        self.assertEqual(m[("Ox Alpha", "GPT-4o-mini")]["D_acq"][:2], (10, 28))
        self.assertEqual(m[("Ox Alpha", "GPT-4o-mini")]["D_contract"][:2], (13, 45))

        # Ox <-> Laguna on 45 common cells
        self.assertEqual(m[("Ox Alpha", "Laguna-S")]["D_total"][:2], (5, 45))
        self.assertEqual(m[("Ox Alpha", "Laguna-S")]["D_acq"][:2], (3, 28))
        self.assertEqual(m[("Ox Alpha", "Laguna-S")]["D_contract"][:2], (14, 45))

        # GPT <-> Laguna on 45 common cells
        self.assertEqual(m[("GPT-4o-mini", "Laguna-S")]["D_total"][:2], (12, 45))
        self.assertEqual(m[("GPT-4o-mini", "Laguna-S")]["D_acq"][:2], (8, 28))
        self.assertEqual(m[("GPT-4o-mini", "Laguna-S")]["D_contract"][:2], (1, 45))


if __name__ == "__main__":
    unittest.main()
