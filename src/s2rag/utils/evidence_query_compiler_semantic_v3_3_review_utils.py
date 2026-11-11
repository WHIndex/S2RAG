"""V3.3 review-only protocol for frozen semantic-plan candidates.

This module does not generate or repair plans. It presents an independently
rendered question/plan comparison, explicit variable connectivity, and concrete
PASS/REJECT/UNKNOWN examples to the existing strict V3.1 review validator.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from .evidence_query_compiler_semantic_v3_1_utils import (
    _REVIEW_CHECKS,
    _schema_catalog,
    render_plan_semantics,
    validate_semantic_v3_1_review,
)
from .evidence_query_compiler_semantic_v3_2_utils import _valid_examples
from .evidence_query_compiler_v2_utils import compile_semantic_payload_v2_3


SEMANTIC_V3_3_REVIEW_PROMPT_VERSION = "independent_relation_coverage_review_v3_3"
SEMANTIC_V3_3_REVIEW_CACHE_VERSION = "semantic_v3_3_review_cache_v1"

_REQUIRED_REVIEW_FIELDS = (
    "decision",
    "answer_target",
    "independent_requirements",
    "step_reviews",
    "uncovered_question_requirements",
    "checks",
    "reasons",
)


def _compile_example(question: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    _, plan, reason = compile_semantic_payload_v2_3(
        question, payload, allow_external_plan_review=True
    )
    if not plan or reason != "validated":
        raise ValueError(f"invalid V3.3 review example: {reason}")
    return plan


def build_omitted_terminal_control() -> tuple[str, Dict[str, Any]]:
    question = "Where did the director of Film Q study?"
    payload = {
        "abstain": False,
        "start_entities": [{
            "mention": "Film Q", "variable": "$film", "semantic_type": "work",
        }],
        "relations": [{
            "relation_cues": ["director"],
            "predicate_family": "direction",
            "predicate_subtype": "",
            "canonical_relation": "",
            "input_variable": "$film",
            "output_variable": "$director",
            "input_semantic_role": "work",
            "output_semantic_role": "director",
            "semantic_direction": "canonical",
            "output_entity_type": "person",
        }],
        "question_requirements": [{
            "requirement_id": "q0", "covered_by_relation_index": 0,
            "relation_cues": ["director"],
            "expected_output_type": "person", "answer_category": "",
        }],
        "answer_operation": "project",
        "final_answer_type": "person",
        "final_answer_category": "",
        "final_answer_variable": "$director",
    }
    return question, _compile_example(question, payload)


def build_relation_substitution_control() -> tuple[str, Dict[str, Any]]:
    """Build a connected plan that substitutes director for producer."""
    question = "Who is the spouse of the producer of Film R?"
    payload = {
        "abstain": False,
        "start_entities": [{
            "mention": "Film R", "variable": "$film", "semantic_type": "work",
        }],
        "relations": [
            {
                "relation_cues": ["producer"],
                "predicate_family": "direction",
                "predicate_subtype": "",
                "canonical_relation": "",
                "input_variable": "$film",
                "output_variable": "$director",
                "input_semantic_role": "work",
                "output_semantic_role": "director",
                "semantic_direction": "canonical",
                "output_entity_type": "person",
            },
            {
                "relation_cues": ["spouse"],
                "predicate_family": "spouse_of",
                "predicate_subtype": "",
                "canonical_relation": "spouse_of",
                "input_variable": "$director",
                "output_variable": "$spouse",
                "input_semantic_role": "spouse",
                "output_semantic_role": "spouse",
                "semantic_direction": "canonical",
                "output_entity_type": "person",
            },
        ],
        "question_requirements": [
            {
                "requirement_id": "q0", "covered_by_relation_index": 0,
                "relation_cues": ["producer"],
                "expected_output_type": "person", "answer_category": "",
            },
            {
                "requirement_id": "q1", "covered_by_relation_index": 1,
                "relation_cues": ["spouse"],
                "expected_output_type": "person", "answer_category": "",
            },
        ],
        "answer_operation": "project",
        "final_answer_type": "person",
        "final_answer_category": "",
        "final_answer_variable": "$spouse",
    }
    return question, _compile_example(question, payload)


def build_title_unknown_control() -> tuple[str, Dict[str, Any]]:
    question = "What is the birthday of the Duke of Exampleland?"
    payload = json.loads(json.dumps(_valid_examples()[1]["output"]))
    payload["start_entities"][0] = {
        "mention": "Duke of Exampleland",
        "variable": "$duke",
        "semantic_type": "person",
    }
    payload["relations"][0]["input_variable"] = "$duke"
    return question, _compile_example(question, payload)


def _checks(default: str = "PASS") -> dict[str, str]:
    return {name: default for name in _REVIEW_CHECKS}


def _pass_example() -> dict[str, Any]:
    example = _valid_examples()[0]
    question = str(example["question"])
    plan = _compile_example(question, example["output"])
    review = {
        "decision": "PASS",
        "answer_target": {
            "question_span": "study",
            "plan_variable": "$institution",
            "decision": "PASS",
            "reason": "The question asks for the study institution and the plan projects $institution.",
        },
        "independent_requirements": [
            {
                "requirement_id": "independent_q0",
                "question_spans": ["director"],
                "semantic_relation": "film has director",
                "mapped_step_ids": ["relation_0"],
                "decision": "PASS",
                "reason": "The director clause is represented by relation_0.",
            },
            {
                "requirement_id": "independent_q1",
                "question_spans": ["study"],
                "semantic_relation": "person studied at institution",
                "mapped_step_ids": ["relation_1"],
                "decision": "PASS",
                "reason": "The study clause is represented by relation_1.",
            },
        ],
        "step_reviews": [
            {
                "step_id": "relation_0", "question_spans": ["director"],
                "decision": "PASS", "reason": "relation_0 preserves the director relation.",
            },
            {
                "step_id": "relation_1", "question_spans": ["study"],
                "decision": "PASS", "reason": "relation_1 preserves the education relation.",
            },
        ],
        "uncovered_question_requirements": [],
        "checks": _checks(),
        "reasons": [
            "$director is produced by relation_0 and consumed by relation_1, so the two steps form one connected chain."
        ],
    }
    return {"question": question, "candidate_plan": render_plan_semantics(plan), "review": review}


def _omitted_example() -> dict[str, Any]:
    question, plan = build_omitted_terminal_control()
    checks = _checks()
    checks["answer_target_preserved"] = "FAIL"
    checks["required_relations_preserved"] = "FAIL"
    checks["bindings_directions_and_projection_faithful"] = "FAIL"
    review = {
        "decision": "REJECT",
        "answer_target": {
            "question_span": "study", "plan_variable": "$director",
            "decision": "FAIL",
            "reason": "The candidate projects the director, while the question asks where that director studied.",
        },
        "independent_requirements": [
            {
                "requirement_id": "independent_q0", "question_spans": ["director"],
                "semantic_relation": "film has director", "mapped_step_ids": ["relation_0"],
                "decision": "PASS", "reason": "relation_0 covers the director clause.",
            },
            {
                "requirement_id": "independent_q1", "question_spans": ["study"],
                "semantic_relation": "person studied at institution", "mapped_step_ids": [],
                "decision": "FAIL", "reason": "No candidate step covers the study relation.",
            },
        ],
        "step_reviews": [{
            "step_id": "relation_0", "question_spans": ["director"],
            "decision": "PASS", "reason": "This step is grounded but insufficient.",
        }],
        "uncovered_question_requirements": ["study"],
        "checks": checks,
        "reasons": ["The terminal education relation is missing."],
    }
    return {"question": question, "candidate_plan": render_plan_semantics(plan), "review": review}


def _substitution_example() -> dict[str, Any]:
    question, plan = build_relation_substitution_control()
    checks = _checks()
    checks["required_relations_preserved"] = "FAIL"
    checks["no_relation_substitution_for_executability"] = "FAIL"
    checks["bindings_directions_and_projection_faithful"] = "FAIL"
    review = {
        "decision": "REJECT",
        "answer_target": {
            "question_span": "spouse", "plan_variable": "$spouse",
            "decision": "FAIL",
            "reason": "The spouse is reached from a director, not from the requested producer.",
        },
        "independent_requirements": [
            {
                "requirement_id": "independent_q0", "question_spans": ["producer"],
                "semantic_relation": "film has producer", "mapped_step_ids": ["relation_0"],
                "decision": "FAIL", "reason": "relation_0 substitutes direction for production.",
            },
            {
                "requirement_id": "independent_q1", "question_spans": ["spouse"],
                "semantic_relation": "producer has spouse", "mapped_step_ids": ["relation_1"],
                "decision": "FAIL", "reason": "relation_1 consumes the substituted director binding.",
            },
        ],
        "step_reviews": [
            {
                "step_id": "relation_0", "question_spans": ["producer"],
                "decision": "FAIL", "reason": "The predicate family is direction, not production.",
            },
            {
                "step_id": "relation_1", "question_spans": ["spouse"],
                "decision": "FAIL", "reason": "The spouse step depends on the wrong intermediate role.",
            },
        ],
        "uncovered_question_requirements": ["producer"],
        "checks": checks,
        "reasons": ["A connected variable chain can still be semantically wrong when a relation is substituted."],
    }
    return {"question": question, "candidate_plan": render_plan_semantics(plan), "review": review}


def _unknown_title_example() -> dict[str, Any]:
    question, plan = build_title_unknown_control()
    checks = _checks()
    checks["answer_target_preserved"] = "UNKNOWN"
    checks["descriptions_not_promoted_to_known_entities"] = "UNKNOWN"
    checks["bindings_directions_and_projection_faithful"] = "UNKNOWN"
    review = {
        "decision": "UNKNOWN",
        "answer_target": {
            "question_span": "birthday", "plan_variable": "$date",
            "decision": "UNKNOWN",
            "reason": "The birth projection is plausible, but the title may not identify one grounded person.",
        },
        "independent_requirements": [{
            "requirement_id": "independent_q0", "question_spans": ["birthday"],
            "semantic_relation": "person has birth date", "mapped_step_ids": ["relation_0"],
            "decision": "UNKNOWN", "reason": "The relation is present but its input identity is uncertain.",
        }],
        "step_reviews": [{
            "step_id": "relation_0", "question_spans": ["birthday"],
            "decision": "UNKNOWN", "reason": "The step is valid only if the title binds a unique person.",
        }],
        "uncovered_question_requirements": [],
        "checks": checks,
        "reasons": ["A descriptive title must not be silently promoted to a known named instance."],
    }
    return {"question": question, "candidate_plan": render_plan_semantics(plan), "review": review}


def review_worked_examples() -> list[dict[str, Any]]:
    examples = [
        _pass_example(),
        _omitted_example(),
        _substitution_example(),
        _unknown_title_example(),
    ]
    for example in examples:


        plan = example["candidate_plan"]


        if not plan.get("relations"):
            raise ValueError("V3.3 review example has no relations")
    return examples


def plan_variable_connectivity(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rendered = render_plan_semantics(plan)
    relations = list(rendered.get("relations") or [])
    rows = []
    for index in range(len(relations) - 1):
        left = relations[index]
        right = relations[index + 1]
        produced = str(left.get("output_variable") or "")
        consumed = str(right.get("input_variable") or "")
        rows.append({
            "producer_step": str(left.get("step_id") or ""),
            "consumer_step": str(right.get("step_id") or ""),
            "produced_variable": produced,
            "consumed_variable": consumed,
            "connected": bool(produced) and produced == consumed,
        })
    return rows


def build_semantic_v3_3_review_messages(
    question: str,
    plan: Mapping[str, Any],
) -> list[dict[str, str]]:
    system = (
        "Audit the candidate semantic plan against the question; never answer the question. "
        "First independently enumerate every relation-bearing clause and the answer target. "
        "Then map each clause to relation_N steps. A chain is connected when one step's "
        "output_variable exactly equals the next step's input_variable; do not reject such a "
        "chain merely because it has multiple steps. Reject omitted relations and predicate "
        "substitution even when the variable chain is connected. A title, role, WH phrase, or "
        "description is not automatically a unique named start entity; use UNKNOWN or REJECT "
        "when the question alone cannot establish the binding. PASS requires bidirectional "
        "coverage: every question relation maps to a justified step and every step maps to a "
        "question relation. Copy exact question substrings into all question-span fields. "
        "Return exactly one JSON object containing every required top-level field. Do not copy "
        "candidate_plan or endpoint_schema into the response. This is development review, not "
        "Human Gold."
    )
    payload = {
        "protocol": SEMANTIC_V3_3_REVIEW_PROMPT_VERSION,
        "cache_contract": SEMANTIC_V3_3_REVIEW_CACHE_VERSION,
        "required_top_level_fields": list(_REQUIRED_REVIEW_FIELDS),
        "required_check_names": list(_REVIEW_CHECKS),
        "allowed_decisions": ["PASS", "REJECT", "UNKNOWN"],
        "worked_examples": review_worked_examples(),
        "question": str(question or "").strip(),
        "candidate_plan": render_plan_semantics(plan),
        "variable_connectivity": plan_variable_connectivity(plan),
        "endpoint_schema": _schema_catalog(),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def validate_review_example(
    question: str,
    plan: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str:
    _, reason = validate_semantic_v3_1_review(question, plan, payload)
    return reason
