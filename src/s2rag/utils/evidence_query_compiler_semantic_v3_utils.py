"""Rule-first semantic frontend with model-based development admission.

This module intentionally reuses the V2.2 semantic generator and V2.3
deterministic lowerer.  It adds exactly one independent Question--Plan review
after structural validation.  The review never sees retrieval, witnesses,
reader predictions, QA gold, or supporting facts.

Model admission is an explicit development-only boundary.  It is neither a
human label nor a formal semantic proof, and the public executor still needs a
separate runtime flag before such a plan can run.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .evidence_query_compiler_utils import compile_evidence_query_plan
from .evidence_query_compiler_v2_utils import (
    HYBRID_V2_2_CACHE_KEY_VERSION,
    HYBRID_V2_2_PROMPT_VERSION,
    PHYSICAL_LOWERING_V2_3_VERSION,
    PREDICATE_ENDPOINT_SCHEMA,
    PREDICATE_ENDPOINT_SCHEMA_VERSION,
    SEMANTIC_QUERY_IR_V2_3_VERSION,
    build_hybrid_v2_2_messages,
    compile_semantic_payload_v2_3,
    validate_v2_physical_plan,
)
from .late_bound_textual_relation_utils import parse_json_object
from .relation_fact_join_utils import (
    PLAN_COMPLETE,
    attach_logical_query_plan_identity,
)


SEMANTIC_V3_PIPELINE_VERSION = "rule_first_semantic_v3_20261029"
SEMANTIC_V3_REVIEW_PROMPT_VERSION = "question_plan_semantic_review_v1"
SEMANTIC_V3_REVIEW_CACHE_VERSION = "semantic_v3_review_cache_contract_v1"
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
    phase = dict(plan.get("compile_phase_status") or {})
    return bool(phase.get("plan_static_valid"))


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
        "attempt_id": str(attempt_id),
        "purpose": str(purpose),
        "request_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "response_payload": dict(payload),
        "cache_hit": bool(cache_hit),
        "input_tokens": metadata.get("prompt_tokens"),
        "output_tokens": metadata.get("completion_tokens"),
    }


def render_plan_semantics(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Render only semantic content for an independent model review."""
    semantic_ir = dict(plan.get("semantic_query_ir") or {})
    rendered_relations = []
    for index, relation in enumerate(semantic_ir.get("relations") or []):
        row = dict(relation)
        rendered_relations.append({
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
        "relations": rendered_relations,
        "answer": {
            "operation": str(semantic_ir.get("answer_operation") or ""),
            "type": str(semantic_ir.get("final_answer_type") or ""),
            "category": str(semantic_ir.get("final_answer_category") or ""),
            "variable": str(semantic_ir.get("final_answer_variable") or ""),
        },
        "unsupported_requirements": list(
            semantic_ir.get("unsupported_requirements") or []
        ),
    }


def build_semantic_review_messages(
    question: str,
    plan: Mapping[str, Any],
) -> list[dict[str, str]]:
    rendered = render_plan_semantics(plan)
    endpoint_definitions = {
        family: {
            "left_role": str(spec.get("left_role") or ""),
            "right_role": str(spec.get("right_role") or ""),
            "symmetric": bool(spec.get("symmetric")),
        }
        for family, spec in sorted(PREDICATE_ENDPOINT_SCHEMA.items())
    }
    system = (
        "Independently review whether a semantic query plan faithfully expresses "
        "the given question. You did not generate the plan. Do not answer the "
        "question and do not use world knowledge to fill unknown entities. Return "
        "JSON only. Check the answer target, every required relation and qualifier, "
        "whether a descriptive phrase was wrongly promoted to a known instance, "
        "added conditions, variable binding/direction/comparison/projection, and "
        "semantic substitutions made only for executability. Equivalent inverse "
        "access is allowed when endpoint definitions preserve meaning. Use PASS "
        "only when all six checks pass. Use REJECT for a localized contradiction "
        "and UNKNOWN when the mapping cannot be established from the question and "
        "definitions. Cite exact substrings from the question and all reviewed "
        "relation_N step IDs. This is a development review, not Human Gold."
    )
    user_payload = {
        "protocol": SEMANTIC_V3_REVIEW_PROMPT_VERSION,
        "cache_contract": SEMANTIC_V3_REVIEW_CACHE_VERSION,
        "question": str(question or "").strip(),
        "rendered_semantic_plan": rendered,
        "endpoint_definitions": endpoint_definitions,
        "required_output": {
            "decision": "PASS|REJECT|UNKNOWN",
            "question_spans": ["exact question substring"],
            "reviewed_step_ids": ["relation_0"],
            "checks": {name: "PASS|FAIL|UNKNOWN" for name in _REVIEW_CHECKS},
            "reasons": ["localized reason tied to a span and step"],
        },
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def validate_semantic_review(
    question: str,
    plan: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    decision = str(payload.get("decision") or "").strip().upper()
    if decision not in _DECISIONS:
        return {}, "invalid_review_decision"
    reasons = [str(value).strip() for value in payload.get("reasons") or [] if str(value).strip()]
    if not reasons:
        return {}, "review_reason_missing"
    question_text = str(question or "")
    question_folded = question_text.casefold()
    spans = [str(value).strip() for value in payload.get("question_spans") or [] if str(value).strip()]
    if not spans or any(span.casefold() not in question_folded for span in spans):
        return {}, "review_question_span_not_grounded"
    rendered = render_plan_semantics(plan)
    valid_step_ids = {
        str(row.get("step_id") or "") for row in rendered.get("relations") or []
    }
    reviewed_step_ids = {
        str(value).strip() for value in payload.get("reviewed_step_ids") or []
        if str(value).strip()
    }
    if not reviewed_step_ids.issubset(valid_step_ids):
        return {}, "review_step_id_unknown"
    checks = {
        str(key): str(value).strip().upper()
        for key, value in dict(payload.get("checks") or {}).items()
    }
    if set(checks) != set(_REVIEW_CHECKS):
        return {}, "review_checks_incomplete"
    if any(value not in {"PASS", "FAIL", "UNKNOWN"} for value in checks.values()):
        return {}, "review_check_value_invalid"
    if decision == "PASS":
        if checks.values() and any(value != "PASS" for value in checks.values()):
            return {}, "review_pass_with_nonpass_check"
        if reviewed_step_ids != valid_step_ids:
            return {}, "review_pass_did_not_cover_all_steps"
    elif decision == "REJECT" and "FAIL" not in checks.values():
        return {}, "review_reject_without_failed_check"
    elif decision == "UNKNOWN" and "UNKNOWN" not in checks.values():
        return {}, "review_unknown_without_unknown_check"
    return {
        "version": SEMANTIC_V3_REVIEW_PROMPT_VERSION,
        "decision": decision,
        "question_spans": spans,
        "reviewed_step_ids": sorted(reviewed_step_ids),
        "checks": checks,
        "reasons": reasons,
        "is_human_gold": False,
        "uses_gold_answer": False,
        "uses_supporting_facts": False,
        "uses_retrieval_or_execution": False,
    }, "validated"


def _admit_model_reviewed_plan(
    plan: Mapping[str, Any],
    review: Mapping[str, Any],
) -> Dict[str, Any]:
    admitted = dict(plan)
    previous_coverage = dict(admitted.get("plan_coverage") or {})
    admitted["plan_coverage"] = {
        "status": PLAN_COMPLETE,
        "reason": "model_based_question_plan_semantic_review_pass",
        "evidence": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
        "prior_structural_coverage": previous_coverage,
    }
    admitted["plan_coverage_status"] = PLAN_COMPLETE
    admitted["question_semantic_validation"] = {
        "version": "question_semantic_validation_v3",
        "status": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
        "basis": "independent_context_question_plan_model_review",
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
        "version": SEMANTIC_V3_REVIEW_PROMPT_VERSION,
        "admission_source": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
        "experimental_execution_eligible": True,
        "human_reviewed": False,
        "is_human_gold": False,
        "review": dict(review),
    }
    admitted = attach_logical_query_plan_identity(
        admitted,
        compiler_source="llm_semantic_fallback_model_reviewed",
        compiler_mode="rule_first_semantic_v3",
        attempt_id="attempt1",
    )
    return admitted


def compile_rule_first_semantic_v3_query_plan(
    question: str,
    *,
    llm_infer: Optional[Callable[..., Any]],
    model_name: str,
    rule_options: Optional[Mapping[str, Any]] = None,
    generation_max_completion_tokens: int = 700,
    review_max_completion_tokens: int = 700,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compile with Rule first, one semantic generation, and one review."""
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
        "version": SEMANTIC_V3_PIPELINE_VERSION,
        "mode": "rule_first_semantic_v3",
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
            "generation_prompt": HYBRID_V2_2_PROMPT_VERSION,
            "generation_cache": HYBRID_V2_2_CACHE_KEY_VERSION,
            "semantic_ir": SEMANTIC_QUERY_IR_V2_3_VERSION,
            "physical_lowering": PHYSICAL_LOWERING_V2_3_VERSION,
            "predicate_endpoint_schema": PREDICATE_ENDPOINT_SCHEMA_VERSION,
            "review_prompt": SEMANTIC_V3_REVIEW_PROMPT_VERSION,
            "review_cache": SEMANTIC_V3_REVIEW_CACHE_VERSION,
        },
    }
    if llm_infer is None:
        trace["reason"] = "llm_not_available"
        trace["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 6)
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace

    generation_messages = build_hybrid_v2_2_messages(
        question,
        max_completion_tokens=int(generation_max_completion_tokens),
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
            purpose="semantic_generation",
            messages=generation_messages,
            content=content,
            metadata=metadata,
            cache_hit=cache_hit,
            payload=payload,
        ))
        trace["request_count"] = 1
        semantic_ir, candidate_plan, lowering_reason = compile_semantic_payload_v2_3(
            question,
            payload,




            allow_external_plan_review=True,
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

    review_messages = build_semantic_review_messages(question, candidate_plan)
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
            purpose="independent_question_plan_semantic_review",
            messages=review_messages,
            content=content,
            metadata=metadata,
            cache_hit=cache_hit,
            payload=review_payload,
        ))
        trace["request_count"] = 2
        review, review_reason = validate_semantic_review(
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
        "selected_source": "model_based_semantic_admission_dev",
        "selected_plan_id": str(selected.get("plan_id") or ""),
        "admission_source": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
        "reason": "model_based_semantic_review_pass",
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
    })
    selected["query_compiler_trace"] = trace
    return selected, trace
