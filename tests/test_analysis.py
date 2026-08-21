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

    def test_request_plan_contains_every_model_probe_once(self):
        plan = oxford.build_request_plan(123)
        pairs = {(m["id"], p["id"]) for m, p in plan}
        self.assertEqual(len(plan), len(oxford.MODELS) * len(oxford.PROBES))
        self.assertEqual(len(pairs), len(plan))

    def test_demo_analysis_is_complete(self):
        observations = oxford.demo_observations("unit-demo")
        summary = oxford.analyze(observations, demo=True)
        self.assertEqual(summary["requests_failed"], 0)
        glm = next(c for c in summary["comparisons"] if c["other_id"] == "glm-5.2")
        self.assertTrue(glm["constant_offset"])
        self.assertEqual(glm["shape_match_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
