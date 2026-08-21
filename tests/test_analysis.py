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

    def test_fail_closed_tokenizer_on_unknown(self):
        count, err = oxford.count_tokens_local("unknown-tokenizer-xyz", "p01", "test")
        self.assertIsNone(count)
        self.assertIsNotNone(err)
        self.assertIn("Unknown tokenizer", err)

    def test_tiktoken_encoding(self):
        count, err = oxford.count_tokens_local("cl100k-local", "p01", "Hello world")
        self.assertIsNone(err)
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)

    def test_envelope_builders(self):
        payload_a = oxford.build_payload("stealth/ox-alpha", "probe_text", envelope_id="envelope_a_minimal")
        self.assertEqual(payload_a["messages"][0]["content"], "Payload:\nprobe_text")

        payload_b = oxford.build_payload("stealth/ox-alpha", "probe_text", envelope_id="envelope_b_standard")
        self.assertTrue(payload_b["messages"][0]["content"].startswith("Return the single word OK."))

        payload_c = oxford.build_payload("stealth/ox-alpha", "probe_text", envelope_id="envelope_c_system")
        self.assertEqual(len(payload_c["messages"]), 2)
        self.assertEqual(payload_c["messages"][0]["role"], "system")
        self.assertEqual(payload_c["messages"][1]["role"], "user")

    def test_synthetic_probe_generator(self):
        probes = oxford.generate_synthetic_corpus(20, seed=42)
        self.assertEqual(len(probes), 20)
        self.assertTrue(probes[0]["id"].startswith("synth-p00000"))
        self.assertGreater(len(probes[0]["text"]), 10)

    def test_provider_pinning_payload(self):
        payload = oxford.build_payload("stealth/ox-alpha", "test", provider_order=["together", "deepinfra"], allow_fallbacks=False)
        self.assertIn("provider", payload)
        self.assertEqual(payload["provider"]["order"], ["together", "deepinfra"])
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
