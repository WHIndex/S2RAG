"""Stepwise relation-coverage gate for semantic plan review."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Tuple

from .evidence_query_compiler_semantic_v3_4_review_utils import (
    build_v3_4_candidate_facts,
)


SEMANTIC_V3_7_RELATION_PROMPT_VERSION = "stepwise_relation_coverage_gate_v3_7"
SEMANTIC_V3_7_RELATION_CACHE_VERSION = "semantic_v3_7_relation_cache_v1"
_STEP_VALUES = {"PASS", "FAIL", "UNKNOWN"}
_UNCOVERED_VALUES = {"NONE", "PRESENT", "UNKNOWN"}


def _step_key(step_id: str) -> str:
    return f"step_{step_id}"


def build_v3_7_relation_messages(
    question: str, plan: Mapping[str, Any]
) -> list[dict[str, str]]:
    facts = build_v3_4_candidate_facts(plan)
    steps = facts.get("steps") or []
    output_keys = {
        _step_key(str(step.get("step_id") or "")): "PASS|FAIL|UNKNOWN"
        for step in steps
    }
    output_keys["uncovered_question_relation"] = "NONE|PRESENT|UNKNOWN"
    system = (
        "Audit relation coverage only; never answer the question and ignore start-entity "
        "grounding. Return exactly the requested flat JSON keys with no explanation or extra "
        "keys. For each displayed step, PASS when its predicate_family semantically expresses "
        "its exact question_cues in this question, FAIL when the family substitutes a different "
        "relation, and UNKNOWN only when genuinely ambiguous. Canonical family meanings include: "
        "direction covers director/directed by; production covers producer/produced by; education "
        "covers study/studied at/attended/graduated from; spouse_of covers spouse; birth with "
        "birth_date covers birthday/date of birth. Equal adjacent variables form a connected "
        "chain. Set uncovered_question_relation=PRESENT when any relation-bearing clause in the "
        "question has no semantically compatible displayed step; otherwise NONE. This is an "
        "external semantic review, not an independent reference annotation."
    )
    payload = {
        "protocol": SEMANTIC_V3_7_RELATION_PROMPT_VERSION,
        "cache_contract": SEMANTIC_V3_7_RELATION_CACHE_VERSION,
        "question": str(question or "").strip(),
        "candidate_steps": steps,
        "variable_connectivity": facts.get("variable_connectivity") or [],
        "answer": facts.get("answer") or {},
        "required_exact_output_keys": output_keys,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def validate_v3_7_relation_gate(
    plan: Mapping[str, Any], payload: Mapping[str, Any]
) -> Tuple[Dict[str, Any], str]:
    steps = build_v3_4_candidate_facts(plan).get("steps") or []
    expected_step_keys = {
        _step_key(str(step.get("step_id") or "")) for step in steps
    }
    expected_keys = expected_step_keys | {"uncovered_question_relation"}
    if {str(key) for key in payload} != expected_keys:
        return {}, "stepwise_gate_exact_keys_mismatch"
    step_decisions = {
        key: str(payload.get(key) or "").strip().upper()
        for key in sorted(expected_step_keys)
    }
    if any(value not in _STEP_VALUES for value in step_decisions.values()):
        return {}, "stepwise_gate_step_value_invalid"
    uncovered = str(payload.get("uncovered_question_relation") or "").strip().upper()
    if uncovered not in _UNCOVERED_VALUES:
        return {}, "stepwise_gate_uncovered_value_invalid"
    if "FAIL" in step_decisions.values() or uncovered == "PRESENT":
        decision = "REJECT"
    elif "UNKNOWN" in step_decisions.values() or uncovered == "UNKNOWN":
        decision = "UNKNOWN"
    else:
        decision = "PASS"
    return {
        "decision": decision,
        "step_decisions": step_decisions,
        "uncovered_question_relation": uncovered,
    }, "validated"
