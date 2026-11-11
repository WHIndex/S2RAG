"""Factorized relation-fidelity and start-grounding semantic gates."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Tuple

from .evidence_query_compiler_semantic_v3_1_utils import _REVIEW_CHECKS
from .evidence_query_compiler_semantic_v3_4_review_utils import (
    build_v3_4_candidate_facts,
)


SEMANTIC_V3_6_RELATION_PROMPT_VERSION = "flat_relation_fidelity_gate_v3_6"
SEMANTIC_V3_6_RELATION_CACHE_VERSION = "semantic_v3_6_relation_cache_v1"
SEMANTIC_V3_6_GROUNDING_PROMPT_VERSION = "start_entity_grounding_gate_v3_6"
SEMANTIC_V3_6_GROUNDING_CACHE_VERSION = "semantic_v3_6_grounding_cache_v1"

_CHECK_VALUES = {"PASS", "FAIL", "UNKNOWN"}
_GROUNDING_VALUES = {"NAMED_INSTANCE", "DESCRIPTION_OR_ROLE", "UNKNOWN"}


def build_v3_6_relation_messages(
    question: str, plan: Mapping[str, Any]
) -> list[dict[str, str]]:
    system = (
        "Audit only relation and answer fidelity of the displayed candidate facts against "
        "the question. Never answer the question and ignore whether the start mention is a "
        "name, title, or description; a separate gate handles grounding. Return exactly the "
        "six requested flat JSON string fields and no other fields. Values are PASS, FAIL, "
        "or UNKNOWN. Read predicate_family values literally. Equal output_variable and the "
        "next input_variable means a connected chain. FAIL required_relations_preserved when "
        "any relation-bearing question clause is absent. FAIL relation substitution only when "
        "a displayed family represents a different relation, not when a canonical family such "
        "as education covers wording such as study or graduate from. No explanations, arrays, "
        "nested objects, markdown, or additional keys. This is external semantic review, not "
        "Human Gold."
    )
    payload = {
        "protocol": SEMANTIC_V3_6_RELATION_PROMPT_VERSION,
        "cache_contract": SEMANTIC_V3_6_RELATION_CACHE_VERSION,
        "question": str(question or "").strip(),
        "candidate_facts": build_v3_4_candidate_facts(plan),
        "required_exact_output_keys": {
            name: "PASS|FAIL|UNKNOWN" for name in _REVIEW_CHECKS
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def build_v3_6_grounding_messages(
    question: str, plan: Mapping[str, Any]
) -> list[dict[str, str]]:
    facts = build_v3_4_candidate_facts(plan)
    starts = facts.get("start_entities") or []
    system = (
        "Classify only whether the displayed start mention is an explicit name of one entity "
        "or a description whose bearer must first be resolved. Return exactly one JSON field "
        "start_entity_grounding and nothing else. Use NAMED_INSTANCE for proper names and "
        "titles of individual works or organizations, for example Inception, The Tempest, or "
        "University of Oxford. Use DESCRIPTION_OR_ROLE for relational descriptions, jobs, and "
        "noble or office titles whose holder is the intended entity, for example the director "
        "of Inception, the president of a country, or Earl of Exampleland. Use UNKNOWN when the "
        "question wording is insufficient. Do not use world knowledge and do not answer the "
        "question."
    )
    payload = {
        "protocol": SEMANTIC_V3_6_GROUNDING_PROMPT_VERSION,
        "cache_contract": SEMANTIC_V3_6_GROUNDING_CACHE_VERSION,
        "question": str(question or "").strip(),
        "start_entities": starts,
        "required_exact_output": {
            "start_entity_grounding": "NAMED_INSTANCE|DESCRIPTION_OR_ROLE|UNKNOWN"
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def validate_v3_6_relation_gate(payload: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    if {str(key) for key in payload} != set(_REVIEW_CHECKS):
        return {}, "relation_gate_exact_keys_mismatch"
    checks = {
        name: str(payload.get(name) or "").strip().upper()
        for name in _REVIEW_CHECKS
    }
    if any(value not in _CHECK_VALUES for value in checks.values()):
        return {}, "relation_gate_value_invalid"
    if "FAIL" in checks.values():
        decision = "REJECT"
    elif "UNKNOWN" in checks.values():
        decision = "UNKNOWN"
    else:
        decision = "PASS"
    return {"checks": checks, "decision": decision}, "validated"


def validate_v3_6_grounding_gate(payload: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    if {str(key) for key in payload} != {"start_entity_grounding"}:
        return {}, "grounding_gate_exact_keys_mismatch"
    grounding = str(payload.get("start_entity_grounding") or "").strip().upper()
    if grounding not in _GROUNDING_VALUES:
        return {}, "grounding_gate_value_invalid"
    decision = {
        "NAMED_INSTANCE": "PASS",
        "DESCRIPTION_OR_ROLE": "REJECT",
        "UNKNOWN": "UNKNOWN",
    }[grounding]
    return {"start_entity_grounding": grounding, "decision": decision}, "validated"


def combine_v3_6_gates(
    relation_gate: Mapping[str, Any], grounding_gate: Mapping[str, Any]
) -> Dict[str, Any]:
    decisions = {
        str(relation_gate.get("decision") or ""),
        str(grounding_gate.get("decision") or ""),
    }
    if "REJECT" in decisions:
        decision = "REJECT"
    elif "UNKNOWN" in decisions:
        decision = "UNKNOWN"
    elif decisions == {"PASS"}:
        decision = "PASS"
    else:
        decision = "INVALID"
    return {
        "version": "factorized_semantic_gate_v3_6",
        "decision": decision,
        "relation_gate": dict(relation_gate),
        "grounding_gate": dict(grounding_gate),
        "auto_execution_eligible": False,
        "experimental_execution_eligible": decision == "PASS",
        "human_reviewed": False,
        "is_human_gold": False,
        "uses_gold_answer": False,
        "uses_supporting_facts": False,
        "uses_retrieval_or_execution": False,
    }
