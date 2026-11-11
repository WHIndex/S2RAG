"""Flat semantic invariant gate for query-plan development admission.

V3.5 intentionally avoids nested model-authored certificates. The model emits
six frozen semantic invariant decisions plus one start-entity grounding class.
Program code validates the exact key set and derives the global gate decision.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Tuple

from .evidence_query_compiler_semantic_v3_1_utils import _REVIEW_CHECKS
from .evidence_query_compiler_semantic_v3_4_review_utils import (
    build_v3_4_candidate_facts,
)


SEMANTIC_V3_5_GATE_PROMPT_VERSION = "flat_semantic_invariant_gate_v3_5"
SEMANTIC_V3_5_GATE_CACHE_VERSION = "semantic_v3_5_gate_cache_v1"
START_ENTITY_GROUNDING_KEY = "start_entity_grounding"
_CHECK_VALUES = {"PASS", "FAIL", "UNKNOWN"}
_GROUNDING_VALUES = {"NAMED_INSTANCE", "DESCRIPTION_OR_ROLE", "UNKNOWN"}
_EXPECTED_KEYS = set(_REVIEW_CHECKS) | {START_ENTITY_GROUNDING_KEY}


def build_semantic_v3_5_gate_messages(
    question: str,
    plan: Mapping[str, Any],
) -> list[dict[str, str]]:
    system = (
        "Audit the displayed candidate facts against the question; never answer the "
        "question and do not use external world knowledge. Return exactly seven flat JSON "
        "string fields and no other fields. For the six semantic checks use PASS, FAIL, or "
        "UNKNOWN. Use FAIL when a question relation is omitted or substituted even if the "
        "plan remains executable. Read displayed predicate_family values literally. Equal "
        "output_variable and next input_variable means a connected chain. For "
        "start_entity_grounding use NAMED_INSTANCE only when the start mention is an explicit "
        "name of one entity in the question; use DESCRIPTION_OR_ROLE for titles, occupations, "
        "roles, or relational descriptions whose bearer must first be resolved; otherwise "
        "use UNKNOWN. No explanations, arrays, nested objects, markdown, or additional keys. "
        "This is external semantic review, not Human Gold."
    )
    output_keys = {
        name: "PASS|FAIL|UNKNOWN" for name in _REVIEW_CHECKS
    }
    output_keys[START_ENTITY_GROUNDING_KEY] = (
        "NAMED_INSTANCE|DESCRIPTION_OR_ROLE|UNKNOWN"
    )
    payload = {
        "protocol": SEMANTIC_V3_5_GATE_PROMPT_VERSION,
        "cache_contract": SEMANTIC_V3_5_GATE_CACHE_VERSION,
        "question": str(question or "").strip(),
        "candidate_facts": build_v3_4_candidate_facts(plan),
        "required_exact_output_keys": output_keys,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def validate_semantic_v3_5_gate(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    keys = {str(key) for key in payload}
    if keys != _EXPECTED_KEYS:
        return {}, "gate_exact_keys_mismatch"
    checks = {
        name: str(payload.get(name) or "").strip().upper()
        for name in _REVIEW_CHECKS
    }
    if any(value not in _CHECK_VALUES for value in checks.values()):
        return {}, "gate_check_value_invalid"
    grounding = str(payload.get(START_ENTITY_GROUNDING_KEY) or "").strip().upper()
    if grounding not in _GROUNDING_VALUES:
        return {}, "gate_start_grounding_invalid"

    if "FAIL" in checks.values() or grounding == "DESCRIPTION_OR_ROLE":
        decision = "REJECT"
    elif "UNKNOWN" in checks.values() or grounding == "UNKNOWN":
        decision = "UNKNOWN"
    else:
        decision = "PASS"
    return {
        "version": SEMANTIC_V3_5_GATE_PROMPT_VERSION,
        "decision": decision,
        "checks": checks,
        "start_entity_grounding": grounding,
        "auto_execution_eligible": False,
        "experimental_execution_eligible": decision == "PASS",
        "human_reviewed": False,
        "is_human_gold": False,
        "uses_gold_answer": False,
        "uses_supporting_facts": False,
        "uses_retrieval_or_execution": False,
    }, "validated"
