import json
import unittest

from s2rag.utils.evidence_query_compiler_semantic_v3_15_runtime_utils import (
    compile_rule_first_semantic_v3_15_query_plan,
)


RULE_OPTIONS = {
    "enable_bound_completion": True,
    "enable_role_directed_join": True,
    "enable_schema_plan_coverage_v7": True,
    "enable_general_plan_coverage_v8": False,
    "enable_schema_exact_nested_join_v9": False,
}


def _response(payload: dict) -> tuple[str, dict, bool]:
    return json.dumps(payload), {"prompt_tokens": 11, "completion_tokens": 7}, False


class QueryCompilerTest(unittest.TestCase):
    def test_anchor_first_lowering_preserves_relation_chain(self) -> None:
        def infer(messages, **_kwargs):
            protocol = json.loads(messages[-1]["content"])["protocol"]
            if protocol == "semantic_skeleton_generation_v3_12_20261029":
                return _response(
                    {
                        "decision": "PLAN",
                        "start_entity": {
                            "mention": "Young Man Luther's author",
                            "semantic_type": "person",
                        },
                        "steps": [
                            {
                                "relation_cue": "spouse",
                                "predicate_family": "spouse_of",
                                "predicate_subtype": "",
                                "traversal": "LEFT_TO_RIGHT",
                                "output_entity_type": "person",
                            }
                        ],
                        "final_answer_type": "person",
                    }
                )
            if protocol == "stepwise_relation_coverage_gate_v3_7":
                return _response(
                    {
                        "step_relation_0": "PASS",
                        "step_relation_1": "PASS",
                        "uncovered_question_relation": "NONE",
                    }
                )
            if protocol == "start_entity_grounding_gate_v3_9":
                return _response({"start_entity_grounding": "NAMED_INSTANCE"})
            raise AssertionError(protocol)

        plan, trace = compile_rule_first_semantic_v3_15_query_plan(
            "Who is the spouse of Young Man Luther's author?",
            llm_infer=infer,
            model_name="fixture-model",
            rule_options=RULE_OPTIONS,
        )
        starts = plan["semantic_query_ir"]["start_entities"]
        relations = plan["semantic_query_ir"]["relations"]
        self.assertEqual(starts[0]["mention"], "Young Man Luther")
        self.assertEqual(
            [row["predicate_family"] for row in relations],
            ["authorship", "spouse_of"],
        )
        self.assertEqual(
            trace["selected_source"],
            "llm_semantic_skeleton_v3_15",
        )


if __name__ == "__main__":
    unittest.main()
