import unittest

from s2rag.utils.bswc_optimizer_utils import source_action_score


class BSWCPolicyTest(unittest.TestCase):
    def test_probability_policy_uses_support_probability(self) -> None:
        state = {"bswc_open_coverage_keys": ["b0:o0", "b0:o1"]}
        action = {
            "operator": "SOURCE_INSPECT",
            "binding_id": "b0",
            "obligation_id": "o0",
            "predicted_coverage_keys": ["b0:o0", "b0:o1"],
            "source_relevance": 1.0,
        }
        low, _ = source_action_score(
            "bswc_probability_only",
            action,
            {"support_probability": 0.4, "expected_scan_rows": 1},
            state,
        )
        high, _ = source_action_score(
            "bswc_probability_only",
            action,
            {"support_probability": 0.8, "expected_scan_rows": 1000},
            state,
        )
        self.assertEqual(low, 0.4)
        self.assertEqual(high, 0.8)

    def test_gain_cost_prefers_lower_physical_cost(self) -> None:
        state = {"bswc_open_coverage_keys": ["b0:o0"]}
        action = {
            "operator": "SOURCE_INSPECT",
            "binding_id": "b0",
            "obligation_id": "o0",
            "predicted_coverage_keys": ["b0:o0"],
        }
        estimate = {
            "support_probability": 0.8,
            "expected_progress": 0.8,
            "expected_witness_progress": 0.8,
            "binding_compatible_coverage": 1,
        }
        cheap, _ = source_action_score(
            "bswc_adaptive_greedy",
            action,
            {**estimate, "expected_scan_rows": 1},
            state,
        )
        expensive, _ = source_action_score(
            "bswc_adaptive_greedy",
            action,
            {**estimate, "expected_scan_rows": 20},
            state,
        )
        self.assertGreater(cheap, expensive)


if __name__ == "__main__":
    unittest.main()
