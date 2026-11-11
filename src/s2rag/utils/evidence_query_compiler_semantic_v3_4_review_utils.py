"""Typed semantic findings and deterministic review lowering for V3.4.

The model decides semantic mappings only. Program code owns field names,
requirement identifiers, plan variables, check completion, and global decision
consistency before the unchanged V3.1 strict validator is applied.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence, Tuple

from .evidence_query_compiler_semantic_v3_1_utils import (
    _REVIEW_CHECKS,
    render_plan_semantics,
    validate_semantic_v3_1_review,
)
from .evidence_query_compiler_semantic_v3_3_review_utils import (
    plan_variable_connectivity,
)


SEMANTIC_V3_4_REVIEW_PROMPT_VERSION = "typed_semantic_findings_review_v3_4"
SEMANTIC_V3_4_REVIEW_CACHE_VERSION = "semantic_v3_4_review_cache_v1"
_ITEM_DECISIONS = {"PASS", "FAIL", "UNKNOWN"}


def _normalized_decision(value: Any) -> str:
    return str(value or "").strip().upper()


def _grounded_spans(question: str, value: Any) -> tuple[list[str], bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return [], False
    spans = [str(item).strip() for item in value if str(item).strip()]
    folded = str(question or "").casefold()
    return spans, bool(spans) and all(span.casefold() in folded for span in spans)


def build_v3_4_candidate_facts(plan: Mapping[str, Any]) -> Dict[str, Any]:
    rendered = render_plan_semantics(plan)
    return {
        "start_entities": rendered.get("start_entities") or [],
        "steps": rendered.get("relations") or [],
        "answer": rendered.get("answer") or {},
        "variable_connectivity": plan_variable_connectivity(plan),
    }


def build_semantic_v3_4_review_messages(
    question: str,
    plan: Mapping[str, Any],
) -> list[dict[str, str]]:
    system = (
        "Audit whether the displayed candidate facts preserve the question semantics. "
        "Never answer the question and do not use world knowledge. Independently identify "
        "every relation-bearing question clause, then map it to displayed step_id values. "
        "Read each step's displayed predicate_family literally: never claim a different "
        "family than the one shown. Equal output_variable and next input_variable means the "
        "steps are connected. Reject a missing relation or predicate substitution even if "
        "the chain is executable. Use UNKNOWN when a title, role, or description cannot be "
        "established as one grounded named instance from the question alone. Return only the "
        "small typed-findings JSON requested below. There are deliberately no worked answers "
        "to copy. This is external semantic review, not Human Gold."
    )
    contract = {
        "answer_target_span": "exact question substring naming the requested answer relation",
        "answer_target_decision": "PASS|FAIL|UNKNOWN",
        "requirements": [{
            "question_spans": ["exact relation-bearing question substring"],
            "semantic_relation": "short plain-language relation",
            "mapped_step_ids": ["displayed step_id or empty"],
            "decision": "PASS|FAIL|UNKNOWN",
        }],
        "steps": [{
            "step_id": "each displayed step_id exactly once",
            "observed_predicate_family": "copy the family displayed for this step",
            "question_spans": ["exact supporting question substring"],
            "decision": "PASS|FAIL|UNKNOWN",
        }],
        "uncovered_question_requirements": ["exact question substring or empty"],
        "failed_checks": ["zero or more allowed check names"],
        "unknown_checks": ["zero or more allowed check names"],
    }
    payload = {
        "protocol": SEMANTIC_V3_4_REVIEW_PROMPT_VERSION,
        "cache_contract": SEMANTIC_V3_4_REVIEW_CACHE_VERSION,
        "question": str(question or "").strip(),
        "candidate_facts": build_v3_4_candidate_facts(plan),
        "allowed_check_names": list(_REVIEW_CHECKS),
        "output_contract": contract,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def lower_semantic_v3_4_findings(
    question: str,
    plan: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Lower compact model findings into the frozen strict review contract."""
    rendered = render_plan_semantics(plan)
    rendered_steps = {
        str(row.get("step_id") or ""): row for row in rendered.get("relations") or []
    }
    if not rendered_steps:
        return {}, "typed_candidate_has_no_steps"

    answer_span = str(payload.get("answer_target_span") or "").strip()
    if not answer_span or answer_span.casefold() not in str(question or "").casefold():
        return {}, "typed_answer_target_not_grounded"
    answer_decision = _normalized_decision(payload.get("answer_target_decision"))
    if answer_decision not in _ITEM_DECISIONS:
        return {}, "typed_answer_target_decision_invalid"

    raw_requirements = payload.get("requirements")
    if (
        not isinstance(raw_requirements, Sequence)
        or isinstance(raw_requirements, (str, bytes))
        or not raw_requirements
    ):
        return {}, "typed_requirements_missing"
    requirements = []
    for index, raw in enumerate(raw_requirements):
        if not isinstance(raw, Mapping):
            return {}, f"typed_requirement_invalid_{index}"
        spans, grounded = _grounded_spans(question, raw.get("question_spans"))
        if not grounded:
            return {}, f"typed_requirement_span_not_grounded_{index}"
        relation = str(raw.get("semantic_relation") or "").strip()
        decision = _normalized_decision(raw.get("decision"))
        step_ids = sorted({
            str(value).strip() for value in raw.get("mapped_step_ids") or []
            if str(value).strip()
        })
        if not relation:
            return {}, f"typed_requirement_relation_missing_{index}"
        if decision not in _ITEM_DECISIONS:
            return {}, f"typed_requirement_decision_invalid_{index}"
        if not set(step_ids).issubset(rendered_steps):
            return {}, f"typed_requirement_step_unknown_{index}"
        if decision == "PASS" and not step_ids:
            return {}, f"typed_pass_requirement_unmapped_{index}"
        requirements.append({
            "requirement_id": f"independent_q{index}",
            "question_spans": spans,
            "semantic_relation": relation,
            "mapped_step_ids": step_ids,
            "decision": decision,
            "reason": (
                f"Typed reviewer marked {relation} as {decision} for "
                f"{', '.join(step_ids) if step_ids else 'no candidate step'}."
            ),
        })

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        return {}, "typed_steps_missing"
    steps = []
    seen_steps = set()
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, Mapping):
            return {}, f"typed_step_invalid_{index}"
        step_id = str(raw.get("step_id") or "").strip()
        if step_id not in rendered_steps or step_id in seen_steps:
            return {}, f"typed_step_id_invalid_{index}"
        expected_family = str(rendered_steps[step_id].get("predicate_family") or "")
        observed_family = str(raw.get("observed_predicate_family") or "").strip()
        if observed_family != expected_family:
            return {}, f"typed_step_family_copy_mismatch_{step_id}"
        spans, grounded = _grounded_spans(question, raw.get("question_spans"))
        if not grounded:
            return {}, f"typed_step_span_not_grounded_{index}"
        decision = _normalized_decision(raw.get("decision"))
        if decision not in _ITEM_DECISIONS:
            return {}, f"typed_step_decision_invalid_{index}"
        seen_steps.add(step_id)
        steps.append({
            "step_id": step_id,
            "question_spans": spans,
            "decision": decision,
            "reason": (
                f"Typed reviewer marked displayed {observed_family} step as {decision}."
            ),
        })
    if seen_steps != set(rendered_steps):
        return {}, "typed_steps_incomplete"

    uncovered, uncovered_grounded = _grounded_spans(
        question, payload.get("uncovered_question_requirements") or []
    )
    if payload.get("uncovered_question_requirements") and not uncovered_grounded:
        return {}, "typed_uncovered_not_grounded"

    failed_checks = {
        str(value).strip() for value in payload.get("failed_checks") or []
        if str(value).strip()
    }
    unknown_checks = {
        str(value).strip() for value in payload.get("unknown_checks") or []
        if str(value).strip()
    }
    allowed_checks = set(_REVIEW_CHECKS)
    if not failed_checks.issubset(allowed_checks):
        return {}, "typed_failed_check_unknown"
    if not unknown_checks.issubset(allowed_checks):
        return {}, "typed_unknown_check_unknown"
    if failed_checks & unknown_checks:
        return {}, "typed_check_conflict"
    checks = {name: "PASS" for name in _REVIEW_CHECKS}
    for name in failed_checks:
        checks[name] = "FAIL"
    for name in unknown_checks:
        checks[name] = "UNKNOWN"

    all_item_decisions = [answer_decision]
    all_item_decisions.extend(row["decision"] for row in requirements)
    all_item_decisions.extend(row["decision"] for row in steps)
    all_item_decisions.extend(checks.values())
    if "FAIL" in all_item_decisions or uncovered:
        decision = "REJECT"
    elif "UNKNOWN" in all_item_decisions:
        decision = "UNKNOWN"
    else:
        decision = "PASS"

    full_payload = {
        "decision": decision,
        "answer_target": {
            "question_span": answer_span,
            "plan_variable": str((rendered.get("answer") or {}).get("variable") or ""),
            "decision": answer_decision,
            "reason": f"Typed reviewer marked the answer target as {answer_decision}.",
        },
        "independent_requirements": requirements,
        "step_reviews": steps,
        "uncovered_question_requirements": uncovered,
        "checks": checks,
        "reasons": [
            f"Deterministically lowered typed semantic findings to {decision}."
        ],
    }
    review, validation_reason = validate_semantic_v3_1_review(
        question, plan, full_payload
    )
    if validation_reason != "validated":
        return {}, f"typed_post_lowering_{validation_reason}"
    review["typed_findings"] = dict(payload)
    review["typed_lowering_version"] = "deterministic_review_lowering_v1"
    return review, "validated"
