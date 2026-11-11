import unittest

from s2rag.profile import build_s2rag_config


class DefaultProfileTest(unittest.TestCase):
    def test_frozen_profile_enables_core_execution(self) -> None:
        config = build_s2rag_config(
            save_dir="/tmp/s2rag-test",
            llm_name="fixture-model",
            llm_base_url="https://service.example/v1",
        )
        self.assertEqual(config.retrieval_backend, "graph_ppr")
        self.assertTrue(config.enable_paper_evidence_query_executor)
        self.assertEqual(
            config.paper_evidence_query_compiler_mode,
            "rule_first_semantic_v3_15",
        )
        self.assertEqual(
            config.paper_evidence_policy,
            "bswc_rank_consistent_binding_qualified_cover",
        )
        self.assertEqual(config.paper_evidence_verification_mode, "strict")
        self.assertTrue(config.paper_evidence_obligation_admission)
        self.assertFalse(config.paper_evidence_apply_structured_answer)


if __name__ == "__main__":
    unittest.main()
