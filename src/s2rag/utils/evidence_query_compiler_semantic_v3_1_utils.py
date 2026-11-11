"""Conservative V3.1 semantic frontend for development-only plan admission.

V3.1 keeps the V3 execution boundary unchanged but replaces its permissive
generation and review protocols. The generator receives the executable schema
contract, while the reviewer must independently enumerate question relations
before comparing them with the candidate plan.

Neither call receives retrieval results, passages, witnesses, reader outputs,
QA answers, or supporting facts. A model PASS is development evidence only and
still requires the executor's explicit model-admission flag.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .evidence_query_compiler_utils import compile_evidence_query_plan
from .evidence_query_compiler_v2_utils import (
    PHYSICAL_LOWERING_V2_3_VERSION,
    PREDICATE_ENDPOINT_SCHEMA,
    PREDICATE_ENDPOINT_SCHEMA_VERSION,
    SEMANTIC_QUERY_IR_V2_3_VERSION,
    SUPPORTED_ANSWER_OPERATIONS,
    SUPPORTED_ENTITY_TYPES,
    SUPPORTED_SUBTYPES,
    compile_semantic_payload_v2_3,
    validate_v2_physical_plan,
)
from .late_bound_textual_relation_utils import parse_json_object
from .relation_fact_join_utils import PLAN_COMPLETE, attach_logical_query_plan_identity


SEMANTIC_V3_1_PIPELINE_VERSION = "rule_first_semantic_v3_1_20261029"
SEMANTIC_V3_1_GENERATION_PROMPT_VERSION = "semantic_query_generation_v3_1"
SEMANTIC_V3_1_GENERATION_CACHE_VERSION = "semantic_v3_1_generation_cache_v1"
SEMANTIC_V3_1_REVIEW_PROMPT_VERSION = "independent_relation_coverage_review_v2"
SEMANTIC_V3_1_REVIEW_CACHE_VERSION = "semantic_v3_1_review_cache_v1"
MODEL_BASED_SEMANTIC_ADMISSION_DEV = "MODEL_BASED_SEMANTIC_ADMISSION_DEV"

_REVIEW_CHECKS = (
    "answer_target_preserved",
    "required_relations_preserved",
    "descriptions_not_promoted_to_known_entities",
    "no_semantic_conditions_added",
    "bindings_directions_and_projection_faithful",
    "no_relation_substitution_for_executability",
)
_DECISIONS = {"PASS", "REJECT", "UNKNOWN"}


def _valid(plan: Mapping[str, Any]) -> bool:
    return bool((plan.get("compile_phase_status") or {}).get("plan_static_valid"))


def _response_parts(response: Any) -> tuple[str, Dict[str, Any], bool]:
    if not isinstance(response, tuple):
        return str(response or ""), {}, False
    content = str(response[0] or "")
    metadata = dict(response[1] or {}) if len(response) > 1 else {}
    cache_hit = bool(response[2]) if len(response) > 2 else bool(
        metadata.get("cache_hit")
    )
    return content, metadata, cache_hit


def _request_record(
    *,
    attempt_id: str,
    purpose: str,
    messages: Sequence[Mapping[str, str]],
    content: str,
    metadata: Mapping[str, Any],
    cache_hit: bool,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    serialized = json.dumps(
        list(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        "attempt_id": attempt_id,
        "purpose": purpose,
        "request_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "response_payload": dict(payload),
        "cache_hit": bool(cache_hit),
        "input_tokens": metadata.get("prompt_tokens"),
        "output_tokens": metadata.get("completion_tokens"),
    }


def _schema_catalog() -> Dict[str, Any]:
    return {
        family: {
            "left_role": str(spec.get("left_role") or ""),
            "right_role": str(spec.get("right_role") or ""),
            "left_types": list(spec.get("left_types") or ()),
            "right_types": list(spec.get("right_types") or ()),
            "symmetric": bool(spec.get("symmetric")),
        }
        for family, spec in sorted(PREDICATE_ENDPOINT_SCHEMA.items())
    }


def build_semantic_v3_1_generation_messages(
    question: str,
    *,
    max_completion_tokens: int = 900,
) -> list[dict[str, str]]:
    """Build a semantic-only request constrained by the executable schema."""
    system = (
        "Compile the question into semantic query JSON; never answer it. First identify every "
        "relation-bearing clause required to reach the requested answer. Emit exactly one "
        "question_requirement and one relation edge for each required clause. A relation cue "
        "and every start mention must be an exact substring of the question. For project, use "
        "exactly one named instance copied from the question as the start. A description, role, "
        "WH phrase, unknown intermediate, or requested answer is a variable, never a start. "
        "Relations must form one connected chain of at most four edges. Use only the supplied "
        "families, endpoint roles, endpoint types, answer operations, entity types, and subtypes. "
        "For canonical direction, input_role=left_role and output_role=right_role; for inverse, "
        "swap them. Do not replace an unsupported relation with a vaguely related executable "
        "family. Do not omit a relation merely because a shorter chain is executable. Do not "
        "emit tuple sides, physical slots, operators, join keys, unknown entity values, answers, "
        "gold fields, or supporting facts. If the complete question cannot be represented by "
        "this contract, return abstain=true with an exact unsupported question span and reason. "
        "Return one JSON object only."
    )
    output_contract = {
        "start_entities": [
            {"mention": "exact named question span", "variable": "$semantic_name", "semantic_type": "allowed type"}
        ],
        "relations": [
            {
                "relation_cues": ["exact question span"],
                "predicate_family": "allowed family",
                "predicate_subtype": "optional allowed subtype",
                "canonical_relation": "optional exact kinship family",
                "input_variable": "$semantic_name",
                "output_variable": "$semantic_name",
                "input_semantic_role": "exact endpoint role",
                "output_semantic_role": "exact endpoint role",
                "semantic_direction": "canonical|inverse",
                "output_entity_type": "allowed type",
            }
        ],
        "question_requirements": [
            {
                "requirement_id": "q0",
                "covered_by_relation_index": 0,
                "relation_cues": ["same exact cue as relation"],
                "expected_output_type": "same type as relation output",
                "answer_category": "terminal only, else empty",
            }
        ],
        "answer_operation": "allowed operation",
        "final_answer_type": "allowed type",
        "final_answer_category": "explicit question category or empty",
        "final_answer_variable": "$terminal_semantic_variable",
    }
    user_payload = {
        "protocol": SEMANTIC_V3_1_GENERATION_PROMPT_VERSION,
        "cache_contract": SEMANTIC_V3_1_GENERATION_CACHE_VERSION,
        "semantic_ir": SEMANTIC_QUERY_IR_V2_3_VERSION,
        "physical_lowering": PHYSICAL_LOWERING_V2_3_VERSION,
        "schema_version": PREDICATE_ENDPOINT_SCHEMA_VERSION,
        "max_completion_tokens": int(max_completion_tokens),
        "allowed_answer_operations": sorted(SUPPORTED_ANSWER_OPERATIONS),
        "allowed_entity_types": sorted(SUPPORTED_ENTITY_TYPES),
        "allowed_predicate_subtypes": sorted(SUPPORTED_SUBTYPES),
        "subtype_constraints": {
            "birth": ["birth_name", "birth_date", "birth_place"],
            "death": ["death_cause", "death_date", "death_place"],
            "employment": [
                "occupation_role",
                "employment_organization",
                "employment_location",
                "employment_generic",
            ],
            "distribution": [
                "distribution_actual_release",
                "distribution_intended_release",
                "distribution_premiere",
            ],
        },
        "endpoint_schema": _schema_catalog(),
        "output_contract": output_contract,
        "question": str(question or "").strip(),
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def render_plan_semantics(plan: Mapping[str, Any]) -> Dict[str, Any]:
    semantic_ir = dict(plan.get("semantic_query_ir") or {})
    relations = []
    for index, raw in enumerate(semantic_ir.get("relations") or []):
        row = dict(raw)
        relations.append({
            "step_id": f"relation_{index}",
            "question_cues": list(row.get("relation_cues") or []),
            "predicate_family": str(row.get("predicate_family") or ""),
            "predicate_subtype": str(row.get("predicate_subtype") or ""),
            "canonical_relation": str(row.get("canonical_relation") or ""),
            "input_variable": str(row.get("input_variable") or ""),
            "output_variable": str(row.get("output_variable") or ""),
            "input_semantic_role": str(row.get("input_semantic_role") or ""),
            "output_semantic_role": str(row.get("output_semantic_role") or ""),
            "semantic_direction": str(row.get("semantic_direction") or ""),
            "output_entity_type": str(row.get("output_entity_type") or ""),
        })
    return {
        "start_entities": [
            {
                "mention": str(row.get("mention") or ""),
                "variable": str(row.get("variable") or ""),
                "semantic_type": str(row.get("semantic_type") or ""),
            }
            for row in semantic_ir.get("start_entities") or []
        ],
        "relations": relations,
        "answer": {
            "operation": str(semantic_ir.get("answer_operation") or ""),
            "type": str(semantic_ir.get("final_answer_type") or ""),
            "category": str(semantic_ir.get("final_answer_category") or ""),
            "variable": str(semantic_ir.get("final_answer_variable") or ""),
        },
    }


def build_semantic_v3_1_review_messages(
    question: str,
    plan: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Require an independent question decomposition before plan comparison."""
    system = (
        "Audit a candidate semantic plan against the question. You did not generate the plan. "
        "Do not answer the question and do not use world knowledge. First independently list "
        "every relation-bearing requirement in the question and the requested answer target. "
        "Then map each independent requirement to candidate relation_N steps. A shorter but "
        "executable path is a REJECT when it omits a question relation. A vaguely related "
        "predicate substitution is REJECT. A descriptive phrase promoted to a known entity is "
        "REJECT. PASS requires complete bidirectional coverage: every independent requirement "
        "maps to a step and every step is justified by a grounded requirement. Every reason, "
        "target, and step review must be nonempty and cite exact question substrings. Use "
        "UNKNOWN when the question alone cannot establish the mapping. Return JSON only. This "
        "is an external semantic review, not Human Gold."
    )
    user_payload = {
        "protocol": SEMANTIC_V3_1_REVIEW_PROMPT_VERSION,
        "cache_contract": SEMANTIC_V3_1_REVIEW_CACHE_VERSION,
        "question": str(question or "").strip(),
        "candidate_plan": render_plan_semantics(plan),
        "endpoint_schema": _schema_catalog(),
        "required_output": {
            "decision": "PASS|REJECT|UNKNOWN",
            "answer_target": {
                "question_span": "exact substring",
                "plan_variable": "$semantic_variable",
                "decision": "PASS|FAIL|UNKNOWN",
                "reason": "nonempty localized reason",
            },
            "independent_requirements": [{
                "requirement_id": "independent_q0",
                "question_spans": ["exact relation-bearing substring"],
                "semantic_relation": "plain-language relation",
                "mapped_step_ids": ["relation_0"],
                "decision": "PASS|FAIL|UNKNOWN",
                "reason": "nonempty mapping reason",
            }],
            "step_reviews": [{
                "step_id": "relation_0",
                "question_spans": ["exact supporting substring"],
                "decision": "PASS|FAIL|UNKNOWN",
                "reason": "nonempty step reason",
            }],
            "uncovered_question_requirements": ["exact unsupported substring or empty"],
            "checks": {name: "PASS|FAIL|UNKNOWN" for name in _REVIEW_CHECKS},
            "reasons": ["at least one nonempty overall reason"],
        },
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def _grounded_spans(question: str, values: Any) -> tuple[list[str], bool]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return [], False
    spans = [str(value).strip() for value in values if str(value).strip()]
    folded = str(question or "").casefold()
    return spans, bool(spans) and all(span.casefold() in folded for span in spans)


def validate_semantic_v3_1_review(
    question: str,
    plan: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Validate the independent decomposition and its complete plan mapping."""
    decision = str(payload.get("decision") or "").strip().upper()
    if decision not in _DECISIONS:
        return {}, "invalid_review_decision"
    reasons = [str(value).strip() for value in payload.get("reasons") or [] if str(value).strip()]
    if not reasons:
        return {}, "review_reason_missing"

    rendered = render_plan_semantics(plan)
    valid_steps = {
        str(row.get("step_id") or "") for row in rendered.get("relations") or []
    }
    if not valid_steps:
        return {}, "review_candidate_has_no_relations"

    answer_target = payload.get("answer_target")
    if not isinstance(answer_target, Mapping):
        return {}, "review_answer_target_missing"
    answer_span = str(answer_target.get("question_span") or "").strip()
    if not answer_span or answer_span.casefold() not in str(question or "").casefold():
        return {}, "review_answer_target_not_grounded"
    target_decision = str(answer_target.get("decision") or "").strip().upper()
    if target_decision not in {"PASS", "FAIL", "UNKNOWN"}:
        return {}, "review_answer_target_decision_invalid"
    target_reason = str(answer_target.get("reason") or "").strip()
    if not target_reason:
        return {}, "review_answer_target_reason_missing"
    expected_answer_variable = str((rendered.get("answer") or {}).get("variable") or "")
    if str(answer_target.get("plan_variable") or "").strip() != expected_answer_variable:
        return {}, "review_answer_target_variable_mismatch"

    raw_requirements = payload.get("independent_requirements")
    if not isinstance(raw_requirements, Sequence) or isinstance(raw_requirements, (str, bytes)) or not raw_requirements:
        return {}, "review_independent_requirements_missing"
    requirements = []
    requirement_ids = set()
    mapped_steps = set()
    for index, raw in enumerate(raw_requirements):
        if not isinstance(raw, Mapping):
            return {}, f"review_requirement_invalid_{index}"
        requirement_id = str(raw.get("requirement_id") or "").strip()
        if not requirement_id or requirement_id in requirement_ids:
            return {}, f"review_requirement_id_invalid_{index}"
        spans, grounded = _grounded_spans(question, raw.get("question_spans"))
        if not grounded:
            return {}, f"review_requirement_span_not_grounded_{index}"
        semantic_relation = str(raw.get("semantic_relation") or "").strip()
        reason = str(raw.get("reason") or "").strip()
        item_decision = str(raw.get("decision") or "").strip().upper()
        if not semantic_relation or not reason:
            return {}, f"review_requirement_explanation_missing_{index}"
        if item_decision not in {"PASS", "FAIL", "UNKNOWN"}:
            return {}, f"review_requirement_decision_invalid_{index}"
        step_ids = {
            str(value).strip() for value in raw.get("mapped_step_ids") or []
            if str(value).strip()
        }
        if not step_ids.issubset(valid_steps):
            return {}, f"review_requirement_step_unknown_{index}"
        if item_decision == "PASS" and not step_ids:
            return {}, f"review_pass_requirement_unmapped_{index}"
        mapped_steps.update(step_ids)
        requirement_ids.add(requirement_id)
        requirements.append({
            "requirement_id": requirement_id,
            "question_spans": spans,
            "semantic_relation": semantic_relation,
            "mapped_step_ids": sorted(step_ids),
            "decision": item_decision,
            "reason": reason,
        })

    raw_step_reviews = payload.get("step_reviews")
    if not isinstance(raw_step_reviews, Sequence) or isinstance(raw_step_reviews, (str, bytes)):
        return {}, "review_step_reviews_missing"
    step_reviews = []
    reviewed_steps = set()
    for index, raw in enumerate(raw_step_reviews):
        if not isinstance(raw, Mapping):
            return {}, f"review_step_invalid_{index}"
        step_id = str(raw.get("step_id") or "").strip()
        if step_id not in valid_steps or step_id in reviewed_steps:
            return {}, f"review_step_id_invalid_{index}"
        spans, grounded = _grounded_spans(question, raw.get("question_spans"))
        if not grounded:
            return {}, f"review_step_span_not_grounded_{index}"
        item_decision = str(raw.get("decision") or "").strip().upper()
        reason = str(raw.get("reason") or "").strip()
        if item_decision not in {"PASS", "FAIL", "UNKNOWN"} or not reason:
            return {}, f"review_step_assessment_invalid_{index}"
        reviewed_steps.add(step_id)
        step_reviews.append({
            "step_id": step_id,
            "question_spans": spans,
            "decision": item_decision,
            "reason": reason,
        })

    uncovered = [
        str(value).strip()
        for value in payload.get("uncovered_question_requirements") or []
        if str(value).strip()
    ]
    if any(value.casefold() not in str(question or "").casefold() for value in uncovered):
        return {}, "review_uncovered_requirement_not_grounded"
    checks = {
        str(key): str(value).strip().upper()
        for key, value in dict(payload.get("checks") or {}).items()
    }
    if set(checks) != set(_REVIEW_CHECKS):
        return {}, "review_checks_incomplete"
    if any(value not in {"PASS", "FAIL", "UNKNOWN"} for value in checks.values()):
        return {}, "review_check_value_invalid"

    requirement_decisions = {row["decision"] for row in requirements}
    step_decisions = {row["decision"] for row in step_reviews}
    if decision == "PASS":
        if target_decision != "PASS" or any(value != "PASS" for value in checks.values()):
            return {}, "review_pass_with_nonpass_check"
        if requirement_decisions != {"PASS"} or step_decisions != {"PASS"}:
            return {}, "review_pass_with_nonpass_mapping"
        if reviewed_steps != valid_steps or mapped_steps != valid_steps:
            return {}, "review_pass_incomplete_bidirectional_coverage"
        if uncovered:
            return {}, "review_pass_with_uncovered_requirement"
    elif decision == "REJECT":
        has_failure = (
            target_decision == "FAIL"
            or "FAIL" in checks.values()
            or "FAIL" in requirement_decisions
            or "FAIL" in step_decisions
            or bool(uncovered)
        )
        if not has_failure:
            return {}, "review_reject_without_failure"
    else:
        has_unknown = (
            target_decision == "UNKNOWN"
            or "UNKNOWN" in checks.values()
            or "UNKNOWN" in requirement_decisions
            or "UNKNOWN" in step_decisions
        )
        if not has_unknown:
            return {}, "review_unknown_without_unknown"

    return {
        "version": SEMANTIC_V3_1_REVIEW_PROMPT_VERSION,
        "decision": decision,
        "answer_target": dict(answer_target),
        "independent_requirements": requirements,
        "step_reviews": step_reviews,
        "uncovered_question_requirements": uncovered,
        "checks": checks,
        "reasons": reasons,
        "is_human_gold": False,
        "uses_gold_answer": False,
        "uses_supporting_facts": False,
        "uses_retrieval_or_execution": False,
    }, "validated"


def _admit_model_reviewed_plan(
    plan: Mapping[str, Any], review: Mapping[str, Any]
) -> Dict[str, Any]:
    admitted = dict(plan)
    prior_coverage = dict(admitted.get("plan_coverage") or {})
    admitted["plan_coverage"] = {
        "status": PLAN_COMPLETE,
        "reason": "independent_relation_coverage_review_pass",
        "evidence": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
        "prior_structural_coverage": prior_coverage,
    }
    admitted["plan_coverage_status"] = PLAN_COMPLETE
    admitted["question_semantic_validation"] = {
        "version": "question_semantic_validation_v3_1",
        "status": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
        "basis": "independent_relation_coverage_model_review",
        "admission_source": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
        "review_decision": "PASS",
        "review_origin": "MODEL_BASED_DEVELOPMENT_ONLY",
        "human_review_status": "NOT_PERFORMED",
        "human_reviewed": False,
        "is_human_gold": False,
        "auto_execution_eligible": False,
        "experimental_execution_eligible": True,
        "review": dict(review),
    }
    admitted["model_based_semantic_admission"] = {
        "version": SEMANTIC_V3_1_REVIEW_PROMPT_VERSION,
        "admission_source": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
        "experimental_execution_eligible": True,
        "human_reviewed": False,
        "is_human_gold": False,
        "review": dict(review),
    }
    return attach_logical_query_plan_identity(
        admitted,
        compiler_source="llm_semantic_v3_1_model_reviewed",
        compiler_mode="rule_first_semantic_v3_1",
        attempt_id="attempt1",
    )


def compile_rule_first_semantic_v3_1_query_plan(
    question: str,
    *,
    llm_infer: Optional[Callable[..., Any]],
    model_name: str,
    rule_options: Optional[Mapping[str, Any]] = None,
    generation_max_completion_tokens: int = 900,
    review_max_completion_tokens: int = 1000,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compile Rule first, then one constrained generation and one review."""
    started = time.perf_counter()
    rule_plan, rule_trace = compile_evidence_query_plan(
        question,
        mode="rule",
        model_name=model_name,
        rule_options=rule_options,
    )
    if _valid(rule_plan):
        return rule_plan, rule_trace

    trace: Dict[str, Any] = {
        "version": SEMANTIC_V3_1_PIPELINE_VERSION,
        "mode": "rule_first_semantic_v3_1",
        "model": str(model_name or ""),
        "rule_trace": rule_trace,
        "selected_source": "rule_fallback",
        "reason": "rule_pattern_unmatched",
        "requests": [],
        "request_count": 0,
        "selection_observed_execution": False,
        "uses_gold_answer": False,
        "uses_supporting_facts": False,
        "uses_retrieval_reader_or_witness": False,
        "versions": {
            "generation_prompt": SEMANTIC_V3_1_GENERATION_PROMPT_VERSION,
            "generation_cache": SEMANTIC_V3_1_GENERATION_CACHE_VERSION,
            "semantic_ir": SEMANTIC_QUERY_IR_V2_3_VERSION,
            "physical_lowering": PHYSICAL_LOWERING_V2_3_VERSION,
            "predicate_endpoint_schema": PREDICATE_ENDPOINT_SCHEMA_VERSION,
            "review_prompt": SEMANTIC_V3_1_REVIEW_PROMPT_VERSION,
            "review_cache": SEMANTIC_V3_1_REVIEW_CACHE_VERSION,
        },
    }
    if llm_infer is None:
        trace["reason"] = "llm_not_available"
        trace["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 6)
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace

    generation_messages = build_semantic_v3_1_generation_messages(
        question, max_completion_tokens=generation_max_completion_tokens
    )
    try:
        response = llm_infer(
            generation_messages,
            max_completion_tokens=max(64, int(generation_max_completion_tokens)),
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content, metadata, cache_hit = _response_parts(response)
        payload = parse_json_object(content)
        if not isinstance(payload, Mapping):
            raise ValueError("invalid_semantic_generation_payload")
        trace["requests"].append(_request_record(
            attempt_id="attempt1",
            purpose="semantic_generation_v3_1",
            messages=generation_messages,
            content=content,
            metadata=metadata,
            cache_hit=cache_hit,
            payload=payload,
        ))
        trace["request_count"] = 1
        semantic_ir, candidate_plan, lowering_reason = compile_semantic_payload_v2_3(
            question, payload, allow_external_plan_review=True
        )
        trace.update({
            "semantic_ir": semantic_ir or None,
            "lowering_reason": str(lowering_reason),
            "structural_candidate_valid": bool(candidate_plan)
            and lowering_reason == "validated",
        })
    except Exception as exc:
        trace.update({
            "reason": "semantic_generation_failed",
            "error_type": type(exc).__name__,
            "error_preview": str(exc)[:240],
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace

    if not candidate_plan or lowering_reason != "validated":
        trace.update({
            "reason": str(lowering_reason),
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace

    review_messages = build_semantic_v3_1_review_messages(question, candidate_plan)
    try:
        response = llm_infer(
            review_messages,
            max_completion_tokens=max(64, int(review_max_completion_tokens)),
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content, metadata, cache_hit = _response_parts(response)
        review_payload = parse_json_object(content)
        if not isinstance(review_payload, Mapping):
            raise ValueError("invalid_semantic_review_payload")
        trace["requests"].append(_request_record(
            attempt_id="attempt2",
            purpose="independent_relation_coverage_review_v2",
            messages=review_messages,
            content=content,
            metadata=metadata,
            cache_hit=cache_hit,
            payload=review_payload,
        ))
        trace["request_count"] = 2
        review, review_reason = validate_semantic_v3_1_review(
            question, candidate_plan, review_payload
        )
        trace.update({
            "semantic_review": review or None,
            "semantic_review_validation_reason": review_reason,
            "semantic_review_decision": str(review.get("decision") or "UNKNOWN"),
        })
    except Exception as exc:
        trace.update({
            "reason": "semantic_review_failed",
            "semantic_review_decision": "UNKNOWN",
            "review_error_type": type(exc).__name__,
            "review_error_preview": str(exc)[:240],
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace

    if not review or str(review.get("decision") or "") != "PASS":
        trace.update({
            "reason": (
                "semantic_review_rejected"
                if str(review.get("decision") or "") == "REJECT"
                else "semantic_review_unknown"
            ),
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace

    selected = _admit_model_reviewed_plan(candidate_plan, review)
    static_validation = validate_v2_physical_plan(selected)
    if not static_validation["valid"]:
        trace.update({
            "reason": f"post_review_{static_validation['reason']}",
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace
    trace.update({
        "selected_source": "model_based_semantic_admission_dev_v3_1",
        "selected_plan_id": str(selected.get("plan_id") or ""),
        "admission_source": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
        "reason": "independent_relation_coverage_review_pass",
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
    })
    selected["query_compiler_trace"] = trace
    return selected, trace
