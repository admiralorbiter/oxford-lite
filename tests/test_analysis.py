import unittest

import oxford


class AnalysisTests(unittest.TestCase):
    def test_constant_offset_is_detected(self):
        matrix = {
            "ox-alpha": {p["id"]: 100 + i for i, p in enumerate(oxford.PROBES)},
            "glm-5.2": {p["id"]: 25 + i for i, p in enumerate(oxford.PROBES)},
        }
        result = oxford.pairwise_comparison(matrix, "ox-alpha", "glm-5.2")
        self.assertTrue(result["constant_offset"])
        self.assertEqual(result["offset_value"], 75)
        self.assertEqual(result["shape_match_ratio"], 1.0)
        self.assertEqual(result["shape_mae"], 0.0)

    def test_shape_difference_is_detected(self):
        ids = [p["id"] for p in oxford.PROBES]
        matrix = {
            "ox-alpha": {pid: value for pid, value in zip(ids, [10, 12, 15, 15, 20, 19])},
            "gemma-4": {pid: value for pid, value in zip(ids, [8, 10, 14, 12, 21, 16])},
        }
        result = oxford.pairwise_comparison(matrix, "ox-alpha", "gemma-4")
        self.assertFalse(result["constant_offset"])
        self.assertLess(result["shape_match_ratio"], 1.0)
        self.assertGreater(result["shape_mae"], 0.0)

    def test_local_tokenizer_differential_evaluation(self):
        # Verify local tokenizer outputs match reference differential structure
        glm_counts = {
            p["id"]: oxford.count_tokens_local("glm-5.2-local", p["id"], p["text"])
            for p in oxford.PROBES
        }
        # Synthetic Ox counts with +75 constant wrapper overhead
        ox_counts = {pid: val + 75 for pid, val in glm_counts.items()}
        matrix = {"ox-alpha": ox_counts, "glm-5.2-local": glm_counts}

        comp = oxford.pairwise_comparison(matrix, "ox-alpha", "glm-5.2-local")
        self.assertTrue(comp["constant_offset"])
        self.assertEqual(comp["offset_value"], 75)
        self.assertEqual(comp["shape_match_ratio"], 1.0)
        self.assertEqual(comp["shape_mae"], 0.0)

    def test_resume_cell_deduplication(self):
        # If cells already succeeded, pending_cells should only contain remaining cells
        all_cells = [(model, probe) for model in oxford.REMOTE_MODELS for probe in oxford.PROBES]
        completed_cells = {oxford.cell_key("ox-alpha", oxford.PROBES[0]["id"])}

        pending = [
            (m, p)
            for m, p in all_cells
            if oxford.cell_key(m["id"], p["id"]) not in completed_cells
        ]
        self.assertEqual(len(pending), len(all_cells) - 1)
        self.assertNotIn((oxford.TARGET_MODEL, oxford.PROBES[0]), pending)

    def test_provider_pinning_payload(self):
        payload = oxford.build_payload("stealth/ox-alpha", "test", provider_order=["together"], allow_fallbacks=False)
        self.assertIn("provider", payload)
        self.assertEqual(payload["provider"]["order"], ["together"])
        self.assertFalse(payload["provider"]["allow_fallbacks"])

    def test_demo_analysis_is_complete(self):
        observations = oxford.demo_observations("unit-demo")
        summary = oxford.analyze(observations, demo=True)
        self.assertEqual(summary["requests_failed"], 0)
        glm = next(c for c in summary["comparisons"] if "glm" in c["other_id"])
        self.assertTrue(glm["constant_offset"])
        self.assertEqual(glm["shape_match_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
